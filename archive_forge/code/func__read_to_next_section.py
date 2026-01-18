import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _read_to_next_section(self):
    section = self._doc.read_to_next_empty_line()
    while not self._is_at_section() and (not self._doc.eof()):
        if not self._doc.peek(-1).strip():
            section += ['']
        section += self._doc.read_to_next_empty_line()
    return section