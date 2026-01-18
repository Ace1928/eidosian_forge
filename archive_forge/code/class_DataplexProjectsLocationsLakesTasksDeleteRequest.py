from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesTasksDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesTasksDeleteRequest object.

  Fields:
    name: Required. The resource name of the task: projects/{project_number}/l
      ocations/{location_id}/lakes/{lake_id}/task/{task_id}.
  """
    name = _messages.StringField(1, required=True)