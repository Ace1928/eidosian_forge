from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDagsDagRunsTaskInstancesGetRequest(_messages.Message):
    """A
  ComposerProjectsLocationsEnvironmentsDagsDagRunsTaskInstancesGetRequest
  object.

  Fields:
    name: Required. The resource name of the task instance to retrieve. Must
      be in the form: "projects/{projectId}/locations/{locationId}/environment
      s/{environmentId}/dags/{dagId}/dagRuns/{dagRunId}/taskInstances/{taskIdW
      ithOptionalMapIndex}".
  """
    name = _messages.StringField(1, required=True)