from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DestinationParallelstore(_messages.Message):
    """Parallelstore as a destination.

  Fields:
    path: Optional. Root directory path to the Paralellstore filesystem,
      starting with '/'. Defaults to '/' if unset.
  """
    path = _messages.StringField(1)