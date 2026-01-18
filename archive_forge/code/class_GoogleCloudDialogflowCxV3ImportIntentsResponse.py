from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ImportIntentsResponse(_messages.Message):
    """The response message for Intents.ImportIntents.

  Fields:
    conflictingResources: Info which resources have conflicts when
      REPORT_CONFLICT merge_option is set in ImportIntentsRequest.
    intents: The unique identifier of the imported intents. Format:
      `projects//locations//agents//intents/`.
  """
    conflictingResources = _messages.MessageField('GoogleCloudDialogflowCxV3ImportIntentsResponseConflictingResources', 1)
    intents = _messages.StringField(2, repeated=True)