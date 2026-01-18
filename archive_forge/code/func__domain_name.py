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
def _domain_name(domain):
    if domain is None:
        return ''
    elif hasattr(domain, 'domain_name'):
        return domain.domain_name()
    elif domain.__class__ is type:
        return domain.__name__
    elif inspect.isfunction(domain):
        return domain.__name__
    else:
        return None