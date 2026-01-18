from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDagsDagRunsTaskInstancesListRequest(_messages.Message):
    """A
  ComposerProjectsLocationsEnvironmentsDagsDagRunsTaskInstancesListRequest
  object.

  Fields:
    filter: An expression for filtering the results. For example:
      executionDate<="2022-02-22T22:22:00Z"
    pageSize: The maximum number of tasks to return.
    pageToken: The next_page_token returned from a previous List request.
    parent: Required. List task instances in the given parent DAG run. Parent
      must be in the form: "projects/{projectId}/locations/{locationId}/enviro
      nments/{environmentId}/dags/{dagId}/dagRuns/{dagRunId}".
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)