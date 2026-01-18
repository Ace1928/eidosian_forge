from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlBackupsListBackupsRequest(_messages.Message):
    """A SqlBackupsListBackupsRequest object.

  Fields:
    filter: Multiple filter queries are space-separated. For example,
      'instance:abc type:FINAL. We allow filters on type, instance name,
      creation time and location.
    pageSize: The maximum number of backups to return per response. The
      service may return fewer than this value. If unspecified, at most 500
      backups are returned. The maximum value is 2000; values above 2000 are
      coerced to 2000.
    pageToken: A page token, received from a previous `ListBackups` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListBackups` must match the call that provided
      the page token.
    parent: Required. The parent, which owns this collection of backups.
      Format: projects/{project}
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)