from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesAssetsCreateRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesAssetsCreateRequest object.

  Fields:
    asset: A Asset resource to be passed as the request body.
    assetId: Required. The ID of the asset resource to be created.
    parent: Required. The parent resource name, in the following form:
      `projects/{project}/locations/{location}/assetTypes/{type}`.
  """
    asset = _messages.MessageField('Asset', 1)
    assetId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)