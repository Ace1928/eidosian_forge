from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HeaderActionRemoveHeader(_messages.Message):
    """The header to remove.

  Fields:
    headerName: Required. The name of the header to remove.
  """
    headerName = _messages.StringField(1)