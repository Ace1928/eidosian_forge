from collections import deque
import os
import typing
from typing import (
from abc import ABC, abstractmethod
from enum import Enum
import string
import copy
import warnings
import re
import sys
from collections.abc import Iterable
import traceback
import types
from operator import itemgetter
from functools import wraps
from threading import RLock
from pathlib import Path
from .util import (
from .exceptions import *
from .actions import *
from .results import ParseResults, _ParseResultsWithOffset
from .unicode import pyparsing_unicode
class _Indent(Empty):

    def __init__(self, ref_col: int):
        super().__init__()
        self.errmsg = f'expected indent at column {ref_col}'
        self.add_condition(lambda s, l, t: col(l, s) == ref_col)