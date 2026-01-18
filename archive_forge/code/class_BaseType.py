from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaseType(_messages.Message):
    """BaseType that describes a service-backed Type.

  Fields:
    collectionOverrides: Allows resource handling overrides for specific
      collections
    credential: Credential used when interacting with this type.
    descriptorUrl: Descriptor Url for the this type.
    options: Options to apply when handling any resources in this service.
  """
    collectionOverrides = _messages.MessageField('CollectionOverride', 1, repeated=True)
    credential = _messages.MessageField('Credential', 2)
    descriptorUrl = _messages.StringField(3)
    options = _messages.MessageField('Options', 4)