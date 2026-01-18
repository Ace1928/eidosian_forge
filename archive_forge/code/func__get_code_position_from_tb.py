import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def _get_code_position_from_tb(tb):
    code, instruction_index = (tb.tb_frame.f_code, tb.tb_lasti)
    return _get_code_position(code, instruction_index)