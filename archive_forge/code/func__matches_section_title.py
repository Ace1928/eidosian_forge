from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _matches_section_title(title, section_title):
    """Returns whether title is a match for a specific section_title.

  Example:
    _matches_section_title('Yields', 'yield') == True

  Args:
    title: The title to check for matching.
    section_title: A specific known section title to check against.
  """
    title = title.lower()
    section_title = section_title.lower()
    return section_title in (title, title[:-1])