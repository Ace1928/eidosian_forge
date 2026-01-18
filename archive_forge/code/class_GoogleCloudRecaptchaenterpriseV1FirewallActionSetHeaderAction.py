from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FirewallActionSetHeaderAction(_messages.Message):
    """A set header action sets a header and forwards the request to the
  backend. This can be used to trigger custom protection implemented on the
  backend.

  Fields:
    key: Optional. The header key to set in the request to the backend server.
    value: Optional. The header value to set in the request to the backend
      server.
  """
    key = _messages.StringField(1)
    value = _messages.StringField(2)