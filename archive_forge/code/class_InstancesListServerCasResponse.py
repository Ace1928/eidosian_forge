from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesListServerCasResponse(_messages.Message):
    """Instances ListServerCas response.

  Fields:
    activeVersion: A string attribute.
    certs: List of server CA certificates for the instance.
    kind: This is always `sql#instancesListServerCas`.
  """
    activeVersion = _messages.StringField(1)
    certs = _messages.MessageField('SslCert', 2, repeated=True)
    kind = _messages.StringField(3)