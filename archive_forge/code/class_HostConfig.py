from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HostConfig(_messages.Message):
    """HostConfig has different instance endpoints.

  Fields:
    api: Output only. API hostname. This is the hostname to use for **Host:
      Data Plane** endpoints.
    gitHttp: Output only. Git HTTP hostname.
    gitSsh: Output only. Git SSH hostname.
    html: Output only. HTML hostname.
  """
    api = _messages.StringField(1)
    gitHttp = _messages.StringField(2)
    gitSsh = _messages.StringField(3)
    html = _messages.StringField(4)