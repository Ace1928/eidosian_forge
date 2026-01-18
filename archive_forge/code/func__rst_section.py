from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _rst_section(line_info):
    """Checks whether the current line is the start of a new RST-style section.

  RST uses directives to specify information. An RST directive, which we refer
  to as a section here, are surrounded with colons. For example, :param name:.

  Args:
    line_info: Information about the current line.
  Returns:
    A Section type if one matches, or None if no section type matches.
  """
    directive = _get_directive(line_info)
    if directive:
        possible_title = directive.split()[0]
        return _section_from_possible_title(possible_title)
    else:
        return None