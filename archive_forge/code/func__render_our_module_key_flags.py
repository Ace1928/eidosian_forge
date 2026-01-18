from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def _render_our_module_key_flags(self, module, output_lines, prefix=''):
    """Returns a help string for the key flags of a given module.

    Args:
      module: module|str, the module to render key flags for.
      output_lines: [str], a list of strings.  The generated help message lines
        will be appended to this list.
      prefix: str, a string that is prepended to each generated help line.
    """
    key_flags = self.get_key_flags_for_module(module)
    if key_flags:
        self._render_module_flags(module, key_flags, output_lines, prefix)