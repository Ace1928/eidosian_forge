from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchProjectsLocationsJobsTaskGroupsTasksGetRequest(_messages.Message):
    """A BatchProjectsLocationsJobsTaskGroupsTasksGetRequest object.

  Fields:
    name: Required. Task name.
  """
    name = _messages.StringField(1, required=True)