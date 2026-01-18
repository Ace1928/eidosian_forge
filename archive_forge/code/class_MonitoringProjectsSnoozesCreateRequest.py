from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsSnoozesCreateRequest(_messages.Message):
    """A MonitoringProjectsSnoozesCreateRequest object.

  Fields:
    parent: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) in which a
      Snooze should be created. The format is: projects/[PROJECT_ID_OR_NUMBER]
    snooze: A Snooze resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    snooze = _messages.MessageField('Snooze', 2)