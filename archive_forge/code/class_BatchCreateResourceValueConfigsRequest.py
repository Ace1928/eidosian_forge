from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateResourceValueConfigsRequest(_messages.Message):
    """Request message to create multiple resource value configs

  Fields:
    requests: Required. The resource value configs to be created.
  """
    requests = _messages.MessageField('CreateResourceValueConfigRequest', 1, repeated=True)