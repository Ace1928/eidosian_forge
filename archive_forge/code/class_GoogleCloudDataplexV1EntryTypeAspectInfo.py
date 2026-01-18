from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EntryTypeAspectInfo(_messages.Message):
    """A GoogleCloudDataplexV1EntryTypeAspectInfo object.

  Fields:
    type: Required aspect type for the entry type.
  """
    type = _messages.StringField(1)