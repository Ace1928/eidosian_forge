from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsFetchDatabasePropertiesRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsFetchDatabasePropertiesRequest
  object.

  Fields:
    environment: Required. The resource name of the environment, in the form:
      "projects/{projectId}/locations/{locationId}/environments/{environmentId
      }"
  """
    environment = _messages.StringField(1, required=True)