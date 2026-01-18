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
class InEnum(object):
    """Domain validation class admitting an enum value/name.

    This will admit any value that is in the specified Enum, including
    Enum members, values, and string names.  The incoming value will be
    automatically cast to an Enum member.

    Parameters
    ----------
    domain: enum.Enum
        The enum that incoming values should be mapped to

    """

    def __init__(self, domain):
        self._domain = domain

    def __call__(self, value):
        try:
            return self._domain(value)
        except ValueError:
            try:
                return self._domain[value]
            except KeyError:
                pass
        raise ValueError('%r is not a valid %s' % (value, self._domain.__name__))

    def domain_name(self):
        return f'InEnum[{self._domain.__name__}]'