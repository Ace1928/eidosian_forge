import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
Previously, Fortran character was incorrectly treated as
    character*1. This hook fixes the usage of the corresponding
    variables in `check`, `dimension`, `=`, and `callstatement`
    expressions.

    The usage of `char*` in `callprotoargument` expression can be left
    unchanged because C `character` is C typedef of `char`, although,
    new implementations should use `character*` in the corresponding
    expressions.

    See https://github.com/numpy/numpy/pull/19388 for more information.

    