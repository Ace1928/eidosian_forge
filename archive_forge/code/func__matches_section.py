from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _matches_section(title, section):
    """Returns whether title is a match any known title for a specific section.

  Example:
    _matches_section_title('Yields', Sections.YIELDS) == True
    _matches_section_title('param', Sections.Args) == True

  Args:
    title: The title to check for matching.
    section: A specific section to check all possible titles for.
  Returns:
    True or False, indicating whether title is a match for the specified
    section.
  """
    for section_title in SECTION_TITLES[section]:
        if _matches_section_title(title, section_title):
            return True
    return False