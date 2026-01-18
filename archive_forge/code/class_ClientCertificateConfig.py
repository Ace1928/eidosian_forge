from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientCertificateConfig(_messages.Message):
    """Configuration for client certificates on the cluster.

  Fields:
    issueClientCertificate: Issue a client certificate.
  """
    issueClientCertificate = _messages.BooleanField(1)