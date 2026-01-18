from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CreatePersistentResourceOperationMetadata(_messages.Message):
    """Details of operations that perform create PersistentResource.

  Fields:
    genericMetadata: Operation metadata for PersistentResource.
    progressMessage: Progress Message for Create LRO
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenericOperationMetadata', 1)
    progressMessage = _messages.StringField(2)