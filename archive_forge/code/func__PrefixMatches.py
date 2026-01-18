from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.console import console_attr
from six.moves import range  # pylint: disable=redefined-builtin
def _PrefixMatches(prefix, possible_matches):
    """Returns the subset of possible_matches that start with prefix.

  Args:
    prefix: str, The prefix to match.
    possible_matches: [str], The list of possible matching strings.

  Returns:
    [str], The subset of possible_matches that start with prefix.
  """
    return [x for x in possible_matches if x.startswith(prefix)]