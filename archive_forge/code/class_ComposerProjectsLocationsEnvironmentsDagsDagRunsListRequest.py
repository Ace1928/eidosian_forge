from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDagsDagRunsListRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsDagsDagRunsListRequest object.

  Fields:
    filter: An expression for filtering the results. For example:
      executionDate<="2022-02-22T22:22:00Z"
    pageSize: The maximum number of DAG runs to return.
    pageToken: The next_page_token returned from a previous List request.
    parent: Required. List DAG runs in the given parent resource. Parent must
      be in the form: "projects/{projectId}/locations/{locationId}/environment
      s/{environmentId}/dags/{dagId}".
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)