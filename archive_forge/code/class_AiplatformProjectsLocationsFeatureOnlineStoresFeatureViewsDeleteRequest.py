from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsDeleteRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsDeleteRequest
  object.

  Fields:
    name: Required. The name of the FeatureView to be deleted. Format: `projec
      ts/{project}/locations/{location}/featureOnlineStores/{feature_online_st
      ore}/featureViews/{feature_view}`
  """
    name = _messages.StringField(1, required=True)