from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1AsyncBatchAnnotateImagesResponse(_messages.Message):
    """Response to an async batch image annotation request.

  Fields:
    outputConfig: The output location and metadata from
      AsyncBatchAnnotateImagesRequest.
  """
    outputConfig = _messages.MessageField('GoogleCloudVisionV1p4beta1OutputConfig', 1)