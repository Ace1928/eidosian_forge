from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsTriggersListRequest(_messages.Message):
    """A CloudbuildProjectsLocationsTriggersListRequest object.

  Fields:
    pageSize: Number of results to return in the list.
    pageToken: Token to provide to skip to a particular spot in the list.
    parent: The parent of the collection of `Triggers`. Format:
      `projects/{project}/locations/{location}`
    projectId: Required. ID of the project for which to list BuildTriggers.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    projectId = _messages.StringField(4)