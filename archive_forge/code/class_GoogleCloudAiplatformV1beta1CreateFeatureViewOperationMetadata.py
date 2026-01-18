from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CreateFeatureViewOperationMetadata(_messages.Message):
    """Details of operations that perform create FeatureView.

  Fields:
    genericMetadata: Operation metadata for FeatureView Create.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenericOperationMetadata', 1)