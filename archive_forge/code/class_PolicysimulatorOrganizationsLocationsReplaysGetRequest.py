from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicysimulatorOrganizationsLocationsReplaysGetRequest(_messages.Message):
    """A PolicysimulatorOrganizationsLocationsReplaysGetRequest object.

  Fields:
    name: Required. The name of the Replay to retrieve, in the following
      format: `{projects|folders|organizations}/{resource-
      id}/locations/global/replays/{replay-id}`, where `{resource-id}` is the
      ID of the project, folder, or organization that owns the `Replay`.
      Example: `projects/my-example-
      project/locations/global/replays/506a5f7f-38ce-4d7d-8e03-479ce1833c36`
  """
    name = _messages.StringField(1, required=True)