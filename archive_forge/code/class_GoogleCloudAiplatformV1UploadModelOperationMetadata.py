from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1UploadModelOperationMetadata(_messages.Message):
    """Details of ModelService.UploadModel operation.

  Fields:
    genericMetadata: The common part of the operation metadata.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 1)