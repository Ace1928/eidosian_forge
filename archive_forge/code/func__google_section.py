from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _google_section(line_info):
    """Checks whether the current line is the start of a new Google-style section.

  This docstring is a Google-style docstring. Google-style sections look like
  this:

    Section Name:
      section body goes here

  Args:
    line_info: Information about the current line.
  Returns:
    A Section type if one matches, or None if no section type matches.
  """
    colon_index = line_info.remaining.find(':')
    possible_title = line_info.remaining[:colon_index]
    return _section_from_possible_title(possible_title)