from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StartActivityNotFound(_messages.Message):
    """User provided intent failed to resolve to an activity.

  Fields:
    action: A string attribute.
    uri: A string attribute.
  """
    action = _messages.StringField(1)
    uri = _messages.StringField(2)