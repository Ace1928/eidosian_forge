from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
class NcAbinitHeader(AbinitHeader):
    """The abinit header found in the NC pseudopotential files."""
    _VARS = dict(zatom=(None, _int_from_str), zion=(None, float), pspdat=(None, float), pspcod=(None, int), pspxc=(None, int), lmax=(None, int), lloc=(None, int), r2well=(None, float), mmax=(None, float), rchrg=(0.0, float), fchrg=(0.0, float), qchrg=(0.0, float))

    def __init__(self, summary, **kwargs):
        super().__init__()
        if 'llocal' in kwargs:
            kwargs['lloc'] = kwargs.pop('llocal')
        self.summary = summary.strip()
        for key, desc in NcAbinitHeader._VARS.items():
            default, astype = desc
            value = kwargs.pop(key, None)
            if value is None:
                value = default
                if default is None:
                    raise RuntimeError(f'Attribute {key} must be specified')
            else:
                try:
                    value = astype(value)
                except Exception:
                    raise RuntimeError(f'Conversion Error for key={key!r}, value={value!r}')
            self[key] = value
        if kwargs:
            self.update(kwargs)

    @staticmethod
    def fhi_header(filename, ppdesc):
        """
        Parse the FHI abinit header. Example:

        Troullier-Martins psp for element  Sc        Thu Oct 27 17:33:22 EDT 1994
            21.00000   3.00000    940714                zatom, zion, pspdat
            1    1    2    0      2001    .00000      pspcod,pspxc,lmax,lloc,mmax,r2well
            1.80626423934776     .22824404341771    1.17378968127746   rchrg,fchrg,qchrg
        """
        lines = _read_nlines(filename, 4)
        try:
            header = _dict_from_lines(lines[:4], [0, 3, 6, 3])
        except ValueError:
            header = _dict_from_lines(lines[:3], [0, 3, 6])
        summary = lines[0]
        return NcAbinitHeader(summary, **header)

    @staticmethod
    def hgh_header(filename, ppdesc):
        """
        Parse the HGH abinit header. Example:

        Hartwigsen-Goedecker-Hutter psp for Ne,  from PRB58, 3641 (1998)
            10   8  010605 zatom,zion,pspdat
            3 1   1 0 2001 0  pspcod,pspxc,lmax,lloc,mmax,r2well
        """
        lines = _read_nlines(filename, 3)
        header = _dict_from_lines(lines[:3], [0, 3, 6])
        summary = lines[0]
        return NcAbinitHeader(summary, **header)

    @staticmethod
    def gth_header(filename, ppdesc):
        """
        Parse the GTH abinit header. Example:

        Goedecker-Teter-Hutter  Wed May  8 14:27:44 EDT 1996
        1   1   960508                     zatom,zion,pspdat
        2   1   0    0    2001    0.       pspcod,pspxc,lmax,lloc,mmax,r2well
        0.2000000 -4.0663326  0.6778322 0 0     rloc, c1, c2, c3, c4
        0 0 0                              rs, h1s, h2s
        0 0                                rp, h1p
          1.36 .2   0.6                    rcutoff, rloc
        """
        lines = _read_nlines(filename, 7)
        header = _dict_from_lines(lines[:3], [0, 3, 6])
        summary = lines[0]
        return NcAbinitHeader(summary, **header)

    @staticmethod
    def oncvpsp_header(filename, ppdesc):
        """
        Parse the ONCVPSP abinit header. Example:

        Li    ONCVPSP  r_core=  2.01  3.02
              3.0000      3.0000      140504    zatom,zion,pspd
             8     2     1     4   600     0    pspcod,pspxc,lmax,lloc,mmax,r2well
          5.99000000  0.00000000  0.00000000    rchrg fchrg qchrg
             2     2     0     0     0    nproj
             0                 extension_switch
           0                        -2.5000025868368D+00 -1.2006906995331D+00
             1  0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
             2  1.0000000000000D-02  4.4140499497377D-02  1.9909081701712D-02
        """
        lines = _read_nlines(filename, 6)
        header = _dict_from_lines(lines[:3], [0, 3, 6])
        summary = lines[0]
        header['pspdat'] = header['pspd']
        header.pop('pspd')
        header['extension_switch'] = int(lines[5].split()[0])
        return NcAbinitHeader(summary, **header)

    @staticmethod
    def tm_header(filename, ppdesc):
        """
        Parse the TM abinit header. Example:

        Troullier-Martins psp for element Fm         Thu Oct 27 17:28:39 EDT 1994
        100.00000  14.00000    940714                zatom, zion, pspdat
           1    1    3    0      2001    .00000      pspcod,pspxc,lmax,lloc,mmax,r2well
           0   4.085   6.246    0   2.8786493        l,e99.0,e99.9,nproj,rcpsp
           .00000000    .0000000000    .0000000000    .00000000   rms,ekb1,ekb2,epsatm
           1   3.116   4.632    1   3.4291849        l,e99.0,e99.9,nproj,rcpsp
           .00000000    .0000000000    .0000000000    .00000000   rms,ekb1,ekb2,epsatm
           2   4.557   6.308    1   2.1865358        l,e99.0,e99.9,nproj,rcpsp
           .00000000    .0000000000    .0000000000    .00000000   rms,ekb1,ekb2,epsatm
           3  23.251  29.387    1   2.4776730        l,e99.0,e99.9,nproj,rcpsp
           .00000000    .0000000000    .0000000000    .00000000   rms,ekb1,ekb2,epsatm
           3.62474762267880     .07409391739104    3.07937699839200   rchrg,fchrg,qchrg
        """
        lines = _read_nlines(filename, -1)
        header = []
        for lineno, line in enumerate(lines):
            header.append(line)
            if lineno == 2:
                tokens = line.split()
                _pspcod, _pspxc, lmax, _lloc = map(int, tokens[:4])
                _mmax, _r2well = map(float, tokens[4:6])
                lines = lines[3:]
                break
        projectors = {}
        for idx in range(2 * (lmax + 1)):
            line = lines[idx]
            if idx % 2 == 0:
                proj_info = [line]
            if idx % 2 == 1:
                proj_info.append(line)
                d = _dict_from_lines(proj_info, [5, 4])
                projectors[int(d['l'])] = d
        header.append(lines[idx + 1])
        summary = header[0]
        header = _dict_from_lines(header, [0, 3, 6, 3])
        return NcAbinitHeader(summary, **header)