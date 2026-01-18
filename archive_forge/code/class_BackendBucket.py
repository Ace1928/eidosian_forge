from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendBucket(_messages.Message):
    """Represents a Cloud Storage Bucket resource. This Cloud Storage bucket
  resource is referenced by a URL map of a load balancer. For more
  information, read Backend Buckets.

  Enums:
    CompressionModeValueValuesEnum: Compress text responses using Brotli or
      gzip compression, based on the client's Accept-Encoding header.

  Fields:
    bucketName: Cloud Storage bucket name.
    cdnPolicy: Cloud CDN configuration for this BackendBucket.
    compressionMode: Compress text responses using Brotli or gzip compression,
      based on the client's Accept-Encoding header.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    customResponseHeaders: Headers that the Application Load Balancer should
      add to proxied responses.
    description: An optional textual description of the resource; provided by
      the client when the resource is created.
    edgeSecurityPolicy: [Output Only] The resource URL for the edge security
      policy associated with this backend bucket.
    enableCdn: If true, enable Cloud CDN for this BackendBucket.
    id: [Output Only] Unique identifier for the resource; defined by the
      server.
    kind: Type of the resource.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    selfLink: [Output Only] Server-defined URL for the resource.
  """

    class CompressionModeValueValuesEnum(_messages.Enum):
        """Compress text responses using Brotli or gzip compression, based on the
    client's Accept-Encoding header.

    Values:
      AUTOMATIC: Automatically uses the best compression based on the Accept-
        Encoding header sent by the client.
      DISABLED: Disables compression. Existing compressed responses cached by
        Cloud CDN will not be served to clients.
    """
        AUTOMATIC = 0
        DISABLED = 1
    bucketName = _messages.StringField(1)
    cdnPolicy = _messages.MessageField('BackendBucketCdnPolicy', 2)
    compressionMode = _messages.EnumField('CompressionModeValueValuesEnum', 3)
    creationTimestamp = _messages.StringField(4)
    customResponseHeaders = _messages.StringField(5, repeated=True)
    description = _messages.StringField(6)
    edgeSecurityPolicy = _messages.StringField(7)
    enableCdn = _messages.BooleanField(8)
    id = _messages.IntegerField(9, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(10, default='compute#backendBucket')
    name = _messages.StringField(11)
    selfLink = _messages.StringField(12)