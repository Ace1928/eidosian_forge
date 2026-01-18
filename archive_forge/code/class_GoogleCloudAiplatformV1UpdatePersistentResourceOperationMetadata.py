from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1UpdatePersistentResourceOperationMetadata(_messages.Message):
    """Details of operations that perform update PersistentResource.

  Fields:
    genericMetadata: Operation metadata for PersistentResource.
    progressMessage: Progress Message for Update LRO
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 1)
    progressMessage = _messages.StringField(2)