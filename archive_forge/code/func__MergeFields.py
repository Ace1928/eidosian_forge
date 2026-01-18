from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def _MergeFields(left, right):
    """Merges two fields, favoring right one.

  Args:
    left: First field.
    right: Second field.

  Returns:
    Merged field.
  """
    return right if right is not None else left