from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1SuggestTrialsMetadata(_messages.Message):
    """Metadata field of a google.longrunning.Operation associated with a
  SuggestTrialsRequest.

  Fields:
    clientId: The identifier of the client that is requesting the suggestion.
    createTime: The time operation was submitted.
    study: The name of the study that the trial belongs to.
    suggestionCount: The number of suggestions requested.
  """
    clientId = _messages.StringField(1)
    createTime = _messages.StringField(2)
    study = _messages.StringField(3)
    suggestionCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)