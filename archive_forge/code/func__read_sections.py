import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _read_sections(self):
    while not self._doc.eof():
        data = self._read_to_next_section()
        name = data[0].strip()
        if name.startswith('..'):
            yield (name, data[1:])
        elif len(data) < 2:
            yield StopIteration
        else:
            yield (name, self._strip(data[2:]))