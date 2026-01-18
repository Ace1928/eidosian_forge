from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicysimulatorProjectsLocationsReplaysListRequest(_messages.Message):
    """A PolicysimulatorProjectsLocationsReplaysListRequest object.

  Fields:
    pageSize: The maximum number of Replay objects to return. Defaults to 50.
      The maximum value is 1000; values above 1000 are rounded down to 1000.
    pageToken: A page token, received from a previous Simulator.ListReplays
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to Simulator.ListReplays must match the call
      that provided the page token.
    parent: Required. The parent resource, in the following format:
      `{projects|folders|organizations}/{resource-id}/locations/global`, where
      `{resource-id}` is the ID of the project, folder, or organization that
      owns the Replay. Example: `projects/my-example-project/locations/global`
      Only `Replay` objects that are direct children of the provided parent
      are listed. In other words, `Replay` objects that are children of a
      project will not be included when the parent is a folder of that
      project.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)