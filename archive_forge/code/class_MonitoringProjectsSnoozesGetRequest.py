from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsSnoozesGetRequest(_messages.Message):
    """A MonitoringProjectsSnoozesGetRequest object.

  Fields:
    name: Required. The ID of the Snooze to retrieve. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/snoozes/[SNOOZE_ID]
  """
    name = _messages.StringField(1, required=True)