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
def _int_from_str(string):
    """
    Convert string into integer.

    Raise:
        TypeError if string is not a valid integer
    """
    float_num = float(string)
    int_num = int(float_num)
    if float_num == int_num:
        return int_num
    int_num = np.rint(float_num)
    logger.warning(f'Converting float {float_num} to int {int_num}')
    return int_num