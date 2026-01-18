from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeleteFeatureValuesOperationMetadata(_messages.Message):
    """Details of operations that delete Feature values.

  Fields:
    genericMetadata: Operation metadata for Featurestore delete Features
      values.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 1)