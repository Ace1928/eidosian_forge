from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1p1beta1Folder(_messages.Message):
    """Message that contains the resource name and display name of a folder
  resource.

  Fields:
    resourceFolder: Full resource name of this folder. See:
      https://cloud.google.com/apis/design/resource_names#full_resource_name
    resourceFolderDisplayName: The user defined display name for this folder.
  """
    resourceFolder = _messages.StringField(1)
    resourceFolderDisplayName = _messages.StringField(2)