from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsActionsGetRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsActionsGetRequest object.

  Fields:
    name: Required. The name of the action to retrieve, in the following form:
      `projects/{project}/locations/{location}/assetTypes/{type}/assets/{asset
      }/actions/{action}`.
  """
    name = _messages.StringField(1, required=True)