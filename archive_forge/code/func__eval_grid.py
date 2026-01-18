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
@staticmethod
def _eval_grid(grid_params):
    """
        This function receives a dictionary with the parameters defining the
        radial mesh and returns a `ndarray` with the mesh.
        """
    eq = grid_params.get('eq').replace(' ', '')
    istart, iend = (int(grid_params.get('istart')), int(grid_params.get('iend')))
    indices = list(range(istart, iend + 1))
    if eq == 'r=a*exp(d*i)':
        a, d = (float(grid_params['a']), float(grid_params['d']))
        mesh = [a * np.exp(d * i) for i in indices]
    elif eq == 'r=a*i/(n-i)':
        a, n = (float(grid_params['a']), float(grid_params['n']))
        mesh = [a * i / (n - i) for i in indices]
    elif eq == 'r=a*(exp(d*i)-1)':
        a, d = (float(grid_params['a']), float(grid_params['d']))
        mesh = [a * (np.exp(d * i) - 1.0) for i in indices]
    elif eq == 'r=d*i':
        d = float(grid_params['d'])
        mesh = [d * i for i in indices]
    elif eq == 'r=(i/n+a)^5/a-a^4':
        a, n = (float(grid_params['a']), float(grid_params['n']))
        mesh = [(i / n + a) ** 5 / a - a ** 4 for i in indices]
    else:
        raise ValueError(f'Unknown grid type: {eq}')
    return np.array(mesh)