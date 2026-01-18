from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExportModelOperationMetadata(_messages.Message):
    """Details of ModelService.ExportModel operation.

  Fields:
    genericMetadata: The common part of the operation metadata.
    outputInfo: Output only. Information further describing the output of this
      Model export.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenericOperationMetadata', 1)
    outputInfo = _messages.MessageField('GoogleCloudAiplatformV1beta1ExportModelOperationMetadataOutputInfo', 2)