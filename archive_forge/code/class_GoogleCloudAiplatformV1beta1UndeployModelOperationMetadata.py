from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1UndeployModelOperationMetadata(_messages.Message):
    """Runtime operation information for EndpointService.UndeployModel.

  Fields:
    genericMetadata: The operation generic information.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenericOperationMetadata', 1)