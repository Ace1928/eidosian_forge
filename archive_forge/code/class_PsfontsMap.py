from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import numpy as np
from matplotlib import _api, cbook
class PsfontsMap:
    """
    A psfonts.map formatted file, mapping TeX fonts to PS fonts.

    Parameters
    ----------
    filename : str or path-like

    Notes
    -----
    For historical reasons, TeX knows many Type-1 fonts by different
    names than the outside world. (For one thing, the names have to
    fit in eight characters.) Also, TeX's native fonts are not Type-1
    but Metafont, which is nontrivial to convert to PostScript except
    as a bitmap. While high-quality conversions to Type-1 format exist
    and are shipped with modern TeX distributions, we need to know
    which Type-1 fonts are the counterparts of which native fonts. For
    these reasons a mapping is needed from internal font names to font
    file names.

    A texmf tree typically includes mapping files called e.g.
    :file:`psfonts.map`, :file:`pdftex.map`, or :file:`dvipdfm.map`.
    The file :file:`psfonts.map` is used by :program:`dvips`,
    :file:`pdftex.map` by :program:`pdfTeX`, and :file:`dvipdfm.map`
    by :program:`dvipdfm`. :file:`psfonts.map` might avoid embedding
    the 35 PostScript fonts (i.e., have no filename for them, as in
    the Times-Bold example above), while the pdf-related files perhaps
    only avoid the "Base 14" pdf fonts. But the user may have
    configured these files differently.

    Examples
    --------
    >>> map = PsfontsMap(find_tex_file('pdftex.map'))
    >>> entry = map[b'ptmbo8r']
    >>> entry.texname
    b'ptmbo8r'
    >>> entry.psname
    b'Times-Bold'
    >>> entry.encoding
    '/usr/local/texlive/2008/texmf-dist/fonts/enc/dvips/base/8r.enc'
    >>> entry.effects
    {'slant': 0.16700000000000001}
    >>> entry.filename
    """
    __slots__ = ('_filename', '_unparsed', '_parsed')

    @lru_cache
    def __new__(cls, filename):
        self = object.__new__(cls)
        self._filename = os.fsdecode(filename)
        with open(filename, 'rb') as file:
            self._unparsed = {}
            for line in file:
                tfmname = line.split(b' ', 1)[0]
                self._unparsed.setdefault(tfmname, []).append(line)
        self._parsed = {}
        return self

    def __getitem__(self, texname):
        assert isinstance(texname, bytes)
        if texname in self._unparsed:
            for line in self._unparsed.pop(texname):
                if self._parse_and_cache_line(line):
                    break
        try:
            return self._parsed[texname]
        except KeyError:
            raise LookupError(f'An associated PostScript font (required by Matplotlib) could not be found for TeX font {texname.decode('ascii')!r} in {self._filename!r}; this problem can often be solved by installing a suitable PostScript font package in your TeX package manager') from None

    def _parse_and_cache_line(self, line):
        """
        Parse a line in the font mapping file.

        The format is (partially) documented at
        http://mirrors.ctan.org/systems/doc/pdftex/manual/pdftex-a.pdf
        https://tug.org/texinfohtml/dvips.html#psfonts_002emap
        Each line can have the following fields:

        - tfmname (first, only required field),
        - psname (defaults to tfmname, must come immediately after tfmname if
          present),
        - fontflags (integer, must come immediately after psname if present,
          ignored by us),
        - special (SlantFont and ExtendFont, only field that is double-quoted),
        - fontfile, encodingfile (optional, prefixed by <, <<, or <[; << always
          precedes a font, <[ always precedes an encoding, < can precede either
          but then an encoding file must have extension .enc; < and << also
          request different font subsetting behaviors but we ignore that; < can
          be separated from the filename by whitespace).

        special, fontfile, and encodingfile can appear in any order.
        """
        if not line or line.startswith((b' ', b'%', b'*', b';', b'#')):
            return
        tfmname = basename = special = encodingfile = fontfile = None
        is_subsetted = is_t1 = is_truetype = False
        matches = re.finditer(b'"([^"]*)(?:"|$)|(\\S+)', line)
        for match in matches:
            quoted, unquoted = match.groups()
            if unquoted:
                if unquoted.startswith(b'<<'):
                    fontfile = unquoted[2:]
                elif unquoted.startswith(b'<['):
                    encodingfile = unquoted[2:]
                elif unquoted.startswith(b'<'):
                    word = unquoted[1:] or next(filter(None, next(matches).groups()))
                    if word.endswith(b'.enc'):
                        encodingfile = word
                    else:
                        fontfile = word
                        is_subsetted = True
                elif tfmname is None:
                    tfmname = unquoted
                elif basename is None:
                    basename = unquoted
            elif quoted:
                special = quoted
        effects = {}
        if special:
            words = reversed(special.split())
            for word in words:
                if word == b'SlantFont':
                    effects['slant'] = float(next(words))
                elif word == b'ExtendFont':
                    effects['extend'] = float(next(words))
        if fontfile is not None:
            if fontfile.endswith((b'.ttf', b'.ttc')):
                is_truetype = True
            elif not fontfile.endswith(b'.otf'):
                is_t1 = True
        elif basename is not None:
            is_t1 = True
        if is_truetype and is_subsetted and (encodingfile is None):
            return
        if not is_t1 and ('slant' in effects or 'extend' in effects):
            return
        if abs(effects.get('slant', 0)) > 1:
            return
        if abs(effects.get('extend', 0)) > 2:
            return
        if basename is None:
            basename = tfmname
        if encodingfile is not None:
            encodingfile = find_tex_file(encodingfile)
        if fontfile is not None:
            fontfile = find_tex_file(fontfile)
        self._parsed[tfmname] = PsFont(texname=tfmname, psname=basename, effects=effects, encoding=encodingfile, filename=fontfile)
        return True