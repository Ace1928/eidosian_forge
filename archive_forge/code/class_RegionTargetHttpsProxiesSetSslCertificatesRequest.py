from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionTargetHttpsProxiesSetSslCertificatesRequest(_messages.Message):
    """A RegionTargetHttpsProxiesSetSslCertificatesRequest object.

  Fields:
    sslCertificates: New set of SslCertificate resources to associate with
      this TargetHttpsProxy resource.
  """
    sslCertificates = _messages.StringField(1, repeated=True)