from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GuestAttributes(_messages.Message):
    """A guest attributes.

  Fields:
    queryPath: The path to be queried. This can be the default namespace ('/')
      or a nested namespace ('/\\/') or a specified key ('/\\/\\')
    queryValue: The value of the requested queried path.
  """
    queryPath = _messages.StringField(1)
    queryValue = _messages.MessageField('GuestAttributesValue', 2)