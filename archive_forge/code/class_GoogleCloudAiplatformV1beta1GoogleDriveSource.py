from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GoogleDriveSource(_messages.Message):
    """The Google Drive location for the input content.

  Fields:
    resourceIds: Required. Google Drive resource IDs.
  """
    resourceIds = _messages.MessageField('GoogleCloudAiplatformV1beta1GoogleDriveSourceResourceId', 1, repeated=True)