from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsEnvironmentsListRequest(_messages.Message):
    """A NotebooksProjectsLocationsEnvironmentsListRequest object.

  Fields:
    pageSize: Maximum return size of the list call.
    pageToken: A previous returned page token that can be used to continue
      listing from the last result.
    parent: Required. Format: `projects/{project_id}/locations/{location}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)