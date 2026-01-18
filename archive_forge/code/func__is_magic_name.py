import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
def _is_magic_name(name):
    return name.value.startswith('__') and name.value.endswith('__')