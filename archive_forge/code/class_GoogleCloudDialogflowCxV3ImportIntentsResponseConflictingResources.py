from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ImportIntentsResponseConflictingResources(_messages.Message):
    """Conflicting resources detected during the import process. Only filled
  when REPORT_CONFLICT is set in the request and there are conflicts in the
  display names.

  Fields:
    entityDisplayNames: Display names of conflicting entities.
    intentDisplayNames: Display names of conflicting intents.
  """
    entityDisplayNames = _messages.StringField(1, repeated=True)
    intentDisplayNames = _messages.StringField(2, repeated=True)