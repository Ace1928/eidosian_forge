from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1TensorboardBlob(_messages.Message):
    """One blob (e.g, image, graph) viewable on a blob metric plot.

  Fields:
    data: Optional. The bytes of the blob is not present unless it's returned
      by the ReadTensorboardBlobData endpoint.
    id: Output only. A URI safe key uniquely identifying a blob. Can be used
      to locate the blob stored in the Cloud Storage bucket of the consumer
      project.
  """
    data = _messages.BytesField(1)
    id = _messages.StringField(2)