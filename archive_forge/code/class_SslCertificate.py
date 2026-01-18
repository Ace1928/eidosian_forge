from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SslCertificate(_messages.Message):
    """Represents an SSL certificate resource. Google Compute Engine has two
  SSL certificate resources: *
  [Global](/compute/docs/reference/rest/beta/sslCertificates) *
  [Regional](/compute/docs/reference/rest/beta/regionSslCertificates) The
  global SSL certificates (sslCertificates) are used by: - Global external
  Application Load Balancers - Classic Application Load Balancers - Proxy
  Network Load Balancers (with target SSL proxies) The regional SSL
  certificates (regionSslCertificates) are used by: - Regional external
  Application Load Balancers - Regional internal Application Load Balancers
  Optionally, certificate file contents that you upload can contain a set of
  up to five PEM-encoded certificates. The API call creates an object
  (sslCertificate) that holds this data. You can use SSL keys and certificates
  to secure connections to a load balancer. For more information, read
  Creating and using SSL certificates, SSL certificates quotas and limits, and
  Troubleshooting SSL certificates.

  Enums:
    TypeValueValuesEnum: (Optional) Specifies the type of SSL certificate,
      either "SELF_MANAGED" or "MANAGED". If not specified, the certificate is
      self-managed and the fields certificate and private_key are used.

  Fields:
    certificate: A value read into memory from a certificate file. The
      certificate file must be in PEM format. The certificate chain must be no
      greater than 5 certs long. The chain must include at least one
      intermediate cert.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    expireTime: [Output Only] Expire time of the certificate. RFC3339
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of the resource. Always compute#sslCertificate
      for SSL certificates.
    managed: Configuration and status of a managed SSL certificate.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    privateKey: A value read into memory from a write-only private key file.
      The private key file must be in PEM format. For security, only insert
      requests include this field.
    region: [Output Only] URL of the region where the regional SSL Certificate
      resides. This field is not applicable to global SSL Certificate.
    selfLink: [Output only] Server-defined URL for the resource.
    selfManaged: Configuration and status of a self-managed SSL certificate.
    subjectAlternativeNames: [Output Only] Domains associated with the
      certificate via Subject Alternative Name.
    type: (Optional) Specifies the type of SSL certificate, either
      "SELF_MANAGED" or "MANAGED". If not specified, the certificate is self-
      managed and the fields certificate and private_key are used.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """(Optional) Specifies the type of SSL certificate, either
    "SELF_MANAGED" or "MANAGED". If not specified, the certificate is self-
    managed and the fields certificate and private_key are used.

    Values:
      MANAGED: Google-managed SSLCertificate.
      SELF_MANAGED: Certificate uploaded by user.
      TYPE_UNSPECIFIED: <no description>
    """
        MANAGED = 0
        SELF_MANAGED = 1
        TYPE_UNSPECIFIED = 2
    certificate = _messages.StringField(1)
    creationTimestamp = _messages.StringField(2)
    description = _messages.StringField(3)
    expireTime = _messages.StringField(4)
    id = _messages.IntegerField(5, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(6, default='compute#sslCertificate')
    managed = _messages.MessageField('SslCertificateManagedSslCertificate', 7)
    name = _messages.StringField(8)
    privateKey = _messages.StringField(9)
    region = _messages.StringField(10)
    selfLink = _messages.StringField(11)
    selfManaged = _messages.MessageField('SslCertificateSelfManagedSslCertificate', 12)
    subjectAlternativeNames = _messages.StringField(13, repeated=True)
    type = _messages.EnumField('TypeValueValuesEnum', 14)