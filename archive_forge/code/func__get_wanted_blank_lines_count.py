import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
def _get_wanted_blank_lines_count(self):
    suite_node = self._indentation_tos.get_latest_suite_node()
    return int(suite_node.parent is None) + 1