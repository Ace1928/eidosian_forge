from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EntrySourceAncestor(_messages.Message):
    """Ancestor contains information about individual items in the hierarchy of
  an Entry.

  Fields:
    name: Optional. The name of the ancestor resource.
    type: Optional. The type of the ancestor resource.
  """
    name = _messages.StringField(1)
    type = _messages.StringField(2)