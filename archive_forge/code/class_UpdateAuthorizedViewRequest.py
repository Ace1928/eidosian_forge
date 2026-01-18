from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateAuthorizedViewRequest(_messages.Message):
    """The request for UpdateAuthorizedView.

  Fields:
    authorizedView: Required. The AuthorizedView to update. The `name` in
      `authorized_view` is used to identify the AuthorizedView. AuthorizedView
      name must in this format: `projects/{project}/instances/{instance}/table
      s/{table}/authorizedViews/{authorized_view}`.
    ignoreWarnings: Optional. If true, ignore the safety checks when updating
      the AuthorizedView.
    updateMask: Optional. The list of fields to update. A mask specifying
      which fields in the AuthorizedView resource should be updated. This mask
      is relative to the AuthorizedView resource, not to the request message.
      A field will be overwritten if it is in the mask. If empty, all fields
      set in the request will be overwritten. A special value `*` means to
      overwrite all fields (including fields not set in the request).
  """
    authorizedView = _messages.MessageField('AuthorizedView', 1)
    ignoreWarnings = _messages.BooleanField(2)
    updateMask = _messages.StringField(3)