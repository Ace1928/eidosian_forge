from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def _join_multiline_values(self):
    defaults = (self.default_section, self._defaults)
    all_sections = itertools.chain((defaults,), self._sections.items())
    for section, options in all_sections:
        for name, val in options.items():
            if isinstance(val, list):
                val = '\n'.join(val).rstrip()
            options[name] = self._interpolation.before_read(self, section, name, val)