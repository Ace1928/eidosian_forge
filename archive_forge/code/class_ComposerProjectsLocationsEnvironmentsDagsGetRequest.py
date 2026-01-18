from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDagsGetRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsDagsGetRequest object.

  Fields:
    name: Required. The resource name of the DAG to retrieve. Must be in the
      form: "projects/{projectId}/locations/{locationId}/environments/{environ
      mentId}/dags/{dagId}".
  """
    name = _messages.StringField(1, required=True)