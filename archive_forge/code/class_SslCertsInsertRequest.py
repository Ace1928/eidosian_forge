from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SslCertsInsertRequest(_messages.Message):
    """SslCerts insert request.

  Fields:
    commonName: User supplied name. Must be a distinct name from the other
      certificates for this instance.
  """
    commonName = _messages.StringField(1)