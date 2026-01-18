from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureGroupsFeaturesGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureGroupsFeaturesGetRequest object.

  Fields:
    name: Required. The name of the Feature resource. Format for entity_type
      as parent: `projects/{project}/locations/{location}/featurestores/{featu
      restore}/entityTypes/{entity_type}` Format for feature_group as parent:
      `projects/{project}/locations/{location}/featureGroups/{feature_group}`
  """
    name = _messages.StringField(1, required=True)