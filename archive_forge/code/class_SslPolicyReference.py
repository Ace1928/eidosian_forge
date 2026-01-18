from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SslPolicyReference(_messages.Message):
    """A SslPolicyReference object.

  Fields:
    sslPolicy: URL of the SSL policy resource. Set this to empty string to
      clear any existing SSL policy associated with the target proxy resource.
  """
    sslPolicy = _messages.StringField(1)