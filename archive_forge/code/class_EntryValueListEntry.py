from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class EntryValueListEntry(_messages.Message):
    """Single entry in a EntryValue.

      Fields:
        entry: A number attribute.
      """
    entry = _messages.FloatField(1, repeated=True)