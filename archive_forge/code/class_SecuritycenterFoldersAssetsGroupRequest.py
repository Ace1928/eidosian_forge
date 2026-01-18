from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersAssetsGroupRequest(_messages.Message):
    """A SecuritycenterFoldersAssetsGroupRequest object.

  Fields:
    groupAssetsRequest: A GroupAssetsRequest resource to be passed as the
      request body.
    parent: Required. The name of the parent to group the assets by. Its
      format is "organizations/[organization_id]", "folders/[folder_id]", or
      "projects/[project_id]".
  """
    groupAssetsRequest = _messages.MessageField('GroupAssetsRequest', 1)
    parent = _messages.StringField(2, required=True)