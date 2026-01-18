from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesGetRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesGetRequest object.

  Fields:
    name: Required. The name of the asset type to retrieve, in the following
      form: `projects/{project}/locations/{location}/assetTypes/{type}`.
  """
    name = _messages.StringField(1, required=True)