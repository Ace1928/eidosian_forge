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
def _dict_from_lines(lines, key_nums, sep=None):
    """
    Helper function to parse formatted text structured like:

    value1 value2 ... sep key1, key2 ...

    key_nums is a list giving the number of keys for each line. 0 if line should be skipped.
    sep is a string denoting the character that separates the keys from the value (None if
    no separator is present).

    Returns:
        dict{key1 : value1, key2 : value2, ...}

    Raises:
        ValueError if parsing fails.
    """
    if isinstance(lines, str):
        lines = [lines]
    if not isinstance(key_nums, collections.abc.Iterable):
        key_nums = list(key_nums)
    if len(lines) != len(key_nums):
        raise ValueError(f'lines = {lines!r}\nkey_nums = {key_nums!r}')
    kwargs = Namespace()
    for idx, nk in enumerate(key_nums):
        if nk == 0:
            continue
        line = lines[idx]
        tokens = [tok.strip() for tok in line.split()]
        values, keys = (tokens[:nk], ''.join(tokens[nk:]))
        keys.replace('[', '').replace(']', '')
        keys = keys.split(',')
        if sep is not None:
            check = keys[0][0]
            if check != sep:
                raise ValueError(f'Expecting separator {sep}, got {check}')
            keys[0] = keys[0][1:]
        if len(values) != len(keys):
            raise ValueError(f'line={line!r}\n len(keys)={len(keys)!r} must equal len(values)={len(values)!r}')
        kwargs.update(zip(keys, values))
    return kwargs