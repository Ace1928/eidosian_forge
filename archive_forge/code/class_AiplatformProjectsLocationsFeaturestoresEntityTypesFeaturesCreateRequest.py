from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesCreateRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesCreateRequest
  object.

  Fields:
    featureId: Required. The ID to use for the Feature, which will become the
      final component of the Feature's resource name. This value may be up to
      128 characters, and valid characters are `[a-z0-9_]`. The first
      character cannot be a number. The value must be unique within an
      EntityType/FeatureGroup.
    googleCloudAiplatformV1Feature: A GoogleCloudAiplatformV1Feature resource
      to be passed as the request body.
    parent: Required. The resource name of the EntityType or FeatureGroup to
      create a Feature. Format for entity_type as parent: `projects/{project}/
      locations/{location}/featurestores/{featurestore}/entityTypes/{entity_ty
      pe}` Format for feature_group as parent:
      `projects/{project}/locations/{location}/featureGroups/{feature_group}`
  """
    featureId = _messages.StringField(1)
    googleCloudAiplatformV1Feature = _messages.MessageField('GoogleCloudAiplatformV1Feature', 2)
    parent = _messages.StringField(3, required=True)