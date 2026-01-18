from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Access(_messages.Message):
    """A GoogleCloudApigeeV1Access object.

  Fields:
    Get: A GoogleCloudApigeeV1AccessGet attribute.
    Remove: A GoogleCloudApigeeV1AccessRemove attribute.
    Set: A GoogleCloudApigeeV1AccessSet attribute.
  """
    Get = _messages.MessageField('GoogleCloudApigeeV1AccessGet', 1)
    Remove = _messages.MessageField('GoogleCloudApigeeV1AccessRemove', 2)
    Set = _messages.MessageField('GoogleCloudApigeeV1AccessSet', 3)