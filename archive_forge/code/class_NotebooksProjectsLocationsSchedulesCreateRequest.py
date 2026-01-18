from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsSchedulesCreateRequest(_messages.Message):
    """A NotebooksProjectsLocationsSchedulesCreateRequest object.

  Fields:
    parent: Required. Format:
      `parent=projects/{project_id}/locations/{location}`
    schedule: A Schedule resource to be passed as the request body.
    scheduleId: Required. User-defined unique ID of this schedule.
  """
    parent = _messages.StringField(1, required=True)
    schedule = _messages.MessageField('Schedule', 2)
    scheduleId = _messages.StringField(3)