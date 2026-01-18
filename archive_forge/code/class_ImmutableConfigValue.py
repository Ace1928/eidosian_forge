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
class ImmutableConfigValue(ConfigValue):

    def __new__(self, *args, **kwds):
        return ConfigValue(*args, **kwds)

    def set_value(self, value):
        if self._cast(value) != self._data:
            raise RuntimeError(str(self) + ' is currently immutable')
        super(ImmutableConfigValue, self).set_value(value)