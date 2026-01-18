from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExportDataOperationMetadata(_messages.Message):
    """Runtime operation information for DatasetService.ExportData.

  Fields:
    gcsOutputDirectory: A Google Cloud Storage directory which path ends with
      '/'. The exported data is stored in the directory.
    genericMetadata: The common part of the operation metadata.
  """
    gcsOutputDirectory = _messages.StringField(1)
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 2)