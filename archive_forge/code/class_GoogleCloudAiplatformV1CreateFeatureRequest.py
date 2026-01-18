from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1CreateFeatureRequest(_messages.Message):
    """Request message for FeaturestoreService.CreateFeature. Request message
  for FeatureRegistryService.CreateFeature.

  Fields:
    feature: Required. The Feature to create.
    featureId: Required. The ID to use for the Feature, which will become the
      final component of the Feature's resource name. This value may be up to
      128 characters, and valid characters are `[a-z0-9_]`. The first
      character cannot be a number. The value must be unique within an
      EntityType/FeatureGroup.
    parent: Required. The resource name of the EntityType or FeatureGroup to
      create a Feature. Format for entity_type as parent: `projects/{project}/
      locations/{location}/featurestores/{featurestore}/entityTypes/{entity_ty
      pe}` Format for feature_group as parent:
      `projects/{project}/locations/{location}/featureGroups/{feature_group}`
  """
    feature = _messages.MessageField('GoogleCloudAiplatformV1Feature', 1)
    featureId = _messages.StringField(2)
    parent = _messages.StringField(3)