from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClientInfoDetail(_messages.Message):
    """Key-value pair of detailed information about the client which invoked
  the test. Examples: {'Version', '1.0'}, {'Release Track', 'BETA'}.

  Fields:
    key: Required. The key of detailed client information.
    value: Required. The value of detailed client information.
  """
    key = _messages.StringField(1)
    value = _messages.StringField(2)