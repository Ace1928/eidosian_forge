from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresDeleteRequest object.

  Fields:
    force: If set to true, any FeatureViews and Features for this
      FeatureOnlineStore will also be deleted. (Otherwise, the request will
      only work if the FeatureOnlineStore has no FeatureViews.)
    name: Required. The name of the FeatureOnlineStore to be deleted. Format:
      `projects/{project}/locations/{location}/featureOnlineStores/{feature_on
      line_store}`
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)