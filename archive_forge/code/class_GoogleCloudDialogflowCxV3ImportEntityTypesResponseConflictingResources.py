from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ImportEntityTypesResponseConflictingResources(_messages.Message):
    """Conflicting resources detected during the import process. Only filled
  when REPORT_CONFLICT is set in the request and there are conflicts in the
  display names.

  Fields:
    entityDisplayNames: Display names of conflicting entities.
    entityTypeDisplayNames: Display names of conflicting entity types.
  """
    entityDisplayNames = _messages.StringField(1, repeated=True)
    entityTypeDisplayNames = _messages.StringField(2, repeated=True)