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
def _get_help_for_modules(self, modules, prefix, include_special_flags):
    """Returns the help string for a list of modules.

    Private to absl.flags package.

    Args:
      modules: List[str], a list of modules to get the help string for.
      prefix: str, a string that is prepended to each generated help line.
      include_special_flags: bool, whether to include description of
        SPECIAL_FLAGS, i.e. --flagfile and --undefok.
    """
    output_lines = []
    for module in modules:
        self._render_our_module_flags(module, output_lines, prefix)
    if include_special_flags:
        self._render_module_flags('absl.flags', six.itervalues(_helpers.SPECIAL_FLAGS._flags()), output_lines, prefix)
    return '\n'.join(output_lines)