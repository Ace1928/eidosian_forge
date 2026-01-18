from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceAttachmentConsumerProjectLimit(_messages.Message):
    """A ServiceAttachmentConsumerProjectLimit object.

  Fields:
    connectionLimit: The value of the limit to set.
    networkUrl: The network URL for the network to set the limit for.
    projectIdOrNum: The project id or number for the project to set the limit
      for.
  """
    connectionLimit = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    networkUrl = _messages.StringField(2)
    projectIdOrNum = _messages.StringField(3)