from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReadTensorboardBlobDataResponse(_messages.Message):
    """Response message for TensorboardService.ReadTensorboardBlobData.

  Fields:
    blobs: Blob messages containing blob bytes.
  """
    blobs = _messages.MessageField('GoogleCloudAiplatformV1beta1TensorboardBlob', 1, repeated=True)