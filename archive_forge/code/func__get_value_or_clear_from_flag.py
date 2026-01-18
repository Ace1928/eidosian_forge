from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.core.util import debug_output
def _get_value_or_clear_from_flag(args, clear_flag, setter_flag):
    """Returns setter value or CLEAR value, prioritizing setter values."""
    value = getattr(args, setter_flag, None)
    if value is not None:
        return value
    if getattr(args, clear_flag, None):
        return CLEAR
    return None