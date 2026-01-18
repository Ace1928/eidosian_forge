from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RenameNfsShareRequest(_messages.Message):
    """Message requesting rename of a server.

  Fields:
    newNfsshareId: Required. The new `id` of the nfsshare.
  """
    newNfsshareId = _messages.StringField(1)