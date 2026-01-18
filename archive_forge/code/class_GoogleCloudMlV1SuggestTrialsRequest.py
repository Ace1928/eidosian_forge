from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1SuggestTrialsRequest(_messages.Message):
    """The request message for the SuggestTrial service method.

  Fields:
    clientId: Required. The identifier of the client that is requesting the
      suggestion. If multiple SuggestTrialsRequests have the same `client_id`,
      the service will return the identical suggested trial if the trial is
      pending, and provide a new trial if the last suggested trial was
      completed.
    suggestionCount: Required. The number of suggestions requested.
  """
    clientId = _messages.StringField(1)
    suggestionCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)