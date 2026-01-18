from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1alphaReplayResult(_messages.Message):
    """The result of replaying a single access tuple against a simulated state.

  Fields:
    accessTuple: The access tuple that was replayed. This field includes
      information about the principal, resource, and permission that were
      involved in the access attempt.
    diff: The difference between the principal's access under the current
      (baseline) policies and the principal's access under the proposed
      (simulated) policies. This field is only included for access tuples that
      were successfully replayed and had different results under the current
      policies and the proposed policies.
    error: The error that caused the access tuple replay to fail. This field
      is only included for access tuples that were not replayed successfully.
    lastSeenDate: The latest date this access tuple was seen in the logs.
    name: The resource name of the `ReplayResult`, in the following format:
      `{projects|folders|organizations}/{resource-
      id}/locations/global/replays/{replay-id}/results/{replay-result-id}`,
      where `{resource-id}` is the ID of the project, folder, or organization
      that owns the Replay. Example: `projects/my-example-project/locations/gl
      obal/replays/506a5f7f-38ce-4d7d-8e03-479ce1833c36/results/1234`
    parent: The Replay that the access tuple was included in.
  """
    accessTuple = _messages.MessageField('GoogleCloudPolicysimulatorV1alphaAccessTuple', 1)
    diff = _messages.MessageField('GoogleCloudPolicysimulatorV1alphaReplayDiff', 2)
    error = _messages.MessageField('GoogleRpcStatus', 3)
    lastSeenDate = _messages.MessageField('GoogleTypeDate', 4)
    name = _messages.StringField(5)
    parent = _messages.StringField(6)