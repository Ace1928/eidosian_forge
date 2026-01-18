from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CollectionOverride(_messages.Message):
    """CollectionOverride allows resource handling overrides for specific
  resources within a BaseType

  Fields:
    collection: The collection that identifies this resource within its
      service.
    methodMap: Custom verb method mappings to support unordered list API
      mappings.
    options: The options to apply to this resource-level override
  """
    collection = _messages.StringField(1)
    methodMap = _messages.MessageField('MethodMap', 2)
    options = _messages.MessageField('Options', 3)