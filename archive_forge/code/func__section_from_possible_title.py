from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _section_from_possible_title(possible_title):
    """Returns a section matched by the possible title, or None if none match.

  Args:
    possible_title: A string that may be the title of a new section.
  Returns:
    A Section type if one matches, or None if no section type matches.
  """
    for section in SECTION_TITLES:
        if _matches_section(possible_title, section):
            return section
    return None