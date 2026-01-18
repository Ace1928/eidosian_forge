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
class _UnpickleableDomain(object):

    def __init__(self, obj):
        self._type = type(obj).__name__
        self._name = obj.name(True)

    def __call__(self, arg):
        logging.error("%s '%s' was pickled with an unpicklable domain.\n    The domain was stripped and lost during the pickle process.  Setting\n    new values on the restored object cannot be mapped into the correct\n    domain.\n" % (self._type, self._name))
        return arg