from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuantityBasedRetention(_messages.Message):
    """A quantity based policy specifies that a certain number of the most
  recent successful backups should be retained.

  Fields:
    count: The number of backups to retain.
  """
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)