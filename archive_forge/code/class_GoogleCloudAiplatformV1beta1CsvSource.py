from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CsvSource(_messages.Message):
    """The storage details for CSV input content.

  Fields:
    gcsSource: Required. Google Cloud Storage location.
  """
    gcsSource = _messages.MessageField('GoogleCloudAiplatformV1beta1GcsSource', 1)