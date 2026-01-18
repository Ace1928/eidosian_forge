import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
def NegativeInt(val):
    """Domain validation function admitting strictly negative integers

    This domain will admit negative integers (n < 0), as well as any
    types that are convertible to negative integers.

    """
    ans = int(val)
    if ans != float(val) or ans >= 0:
        raise ValueError('Expected negative int, but received %s' % (val,))
    return ans