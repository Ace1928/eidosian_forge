from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesAuthorizedViewsDeleteRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesAuthorizedViewsDeleteRequest
  object.

  Fields:
    etag: Optional. The current etag of the AuthorizedView. If an etag is
      provided and does not match the current etag of the AuthorizedView,
      deletion will be blocked and an ABORTED error will be returned.
    name: Required. The unique name of the AuthorizedView to be deleted.
      Values are of the form `projects/{project}/instances/{instance}/tables/{
      table}/authorizedViews/{authorized_view}`.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)