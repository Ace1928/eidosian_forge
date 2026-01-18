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
def _parse_enc(path):
    """
    Parse a \\*.enc file referenced from a psfonts.map style file.

    The format supported by this function is a tiny subset of PostScript.

    Parameters
    ----------
    path : `os.PathLike`

    Returns
    -------
    list
        The nth entry of the list is the PostScript glyph name of the nth
        glyph.
    """
    no_comments = re.sub('%.*', '', Path(path).read_text(encoding='ascii'))
    array = re.search('(?s)\\[(.*)\\]', no_comments).group(1)
    lines = [line for line in array.split() if line]
    if all((line.startswith('/') for line in lines)):
        return [line[1:] for line in lines]
    else:
        raise ValueError(f'Failed to parse {path} as Postscript encoding')