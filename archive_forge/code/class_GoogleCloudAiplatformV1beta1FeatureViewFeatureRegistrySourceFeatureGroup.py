from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureViewFeatureRegistrySourceFeatureGroup(_messages.Message):
    """Features belonging to a single feature group that will be synced to
  Online Store.

  Fields:
    featureGroupId: Required. Identifier of the feature group.
    featureIds: Required. Identifiers of features under the feature group.
  """
    featureGroupId = _messages.StringField(1)
    featureIds = _messages.StringField(2, repeated=True)