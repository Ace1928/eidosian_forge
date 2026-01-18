from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def IsValidLabelKey(key):
    """Implements the PCRE r'[\\p{Ll}\\p{Lo}][\\p{Ll}\\p{Lo}\\p{N}_-]{0,62}'.

  The key must start with a lowercase character and must be a valid label value.

  Args:
    key: The label key, a string.

  Returns:
    True if the key is valid; False if not.
  """
    if not key or not _IsLower(key[0]):
        return False
    return IsValidLabelValue(key)