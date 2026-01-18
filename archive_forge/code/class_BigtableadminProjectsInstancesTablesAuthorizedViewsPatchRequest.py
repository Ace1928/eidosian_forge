from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesAuthorizedViewsPatchRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesAuthorizedViewsPatchRequest
  object.

  Fields:
    authorizedView: A AuthorizedView resource to be passed as the request
      body.
    ignoreWarnings: Optional. If true, ignore the safety checks when updating
      the AuthorizedView.
    name: Identifier. The name of this AuthorizedView. Values are of the form
      `projects/{project}/instances/{instance}/tables/{table}/authorizedViews/
      {authorized_view}`
    updateMask: Optional. The list of fields to update. A mask specifying
      which fields in the AuthorizedView resource should be updated. This mask
      is relative to the AuthorizedView resource, not to the request message.
      A field will be overwritten if it is in the mask. If empty, all fields
      set in the request will be overwritten. A special value `*` means to
      overwrite all fields (including fields not set in the request).
  """
    authorizedView = _messages.MessageField('AuthorizedView', 1)
    ignoreWarnings = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)