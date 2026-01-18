from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ImportEntityTypesResponse(_messages.Message):
    """The response message for EntityTypes.ImportEntityTypes.

  Fields:
    conflictingResources: Info which resources have conflicts when
      REPORT_CONFLICT merge_option is set in ImportEntityTypesRequest.
    entityTypes: The unique identifier of the imported entity types. Format:
      `projects//locations//agents//entity_types/`.
  """
    conflictingResources = _messages.MessageField('GoogleCloudDialogflowCxV3ImportEntityTypesResponseConflictingResources', 1)
    entityTypes = _messages.StringField(2, repeated=True)