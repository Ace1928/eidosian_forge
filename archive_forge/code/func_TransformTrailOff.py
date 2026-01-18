from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformTrailOff(r, character_limit, undefined=''):
    """Returns r if less than limit, else abbreviated r followed by ellipsis.

  Args:
    r: A string.
    character_limit: An int. Max length of return string. Must be greater than 3
      because ellipsis (3 chars) is appended to abridged strings.
    undefined: A string. Return if r or character_limit is invalid.

  Returns:
    r if below character_limit or invalid character_limit
      (non-int or not greater than 3), else abridged r.
  """
    try:
        character_limit_int = int(character_limit)
    except (AttributeError, TypeError, ValueError):
        return undefined
    if character_limit_int <= 3:
        return undefined
    if len(r) < character_limit_int:
        return r
    return r[:character_limit_int - 3] + '...'