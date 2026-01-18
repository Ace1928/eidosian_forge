from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingServer(_messages.Message):
    """Logging server to receive vCenter or ESXi logs.

  Enums:
    ProtocolValueValuesEnum: Required. Protocol used by vCenter to send logs
      to a logging server.
    SourceTypeValueValuesEnum: Required. The type of component that produces
      logs that will be forwarded to this logging server.

  Fields:
    createTime: Output only. Creation time of this resource.
    etag: Optional. Checksum that may be sent on update and delete requests to
      ensure that the user-provided value is up to date before the server
      processes a request. The server computes checksums based on the value of
      other fields in the request.
    hostname: Required. Fully-qualified domain name (FQDN) or IP Address of
      the logging server.
    name: Output only. The resource name of this logging server. Resource
      names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud/loggingServers/my-logging-server`
    port: Required. Port number at which the logging server receives logs.
    protocol: Required. Protocol used by vCenter to send logs to a logging
      server.
    sourceType: Required. The type of component that produces logs that will
      be forwarded to this logging server.
    uid: Output only. System-generated unique identifier for the resource.
    updateTime: Output only. Last update time of this resource.
  """

    class ProtocolValueValuesEnum(_messages.Enum):
        """Required. Protocol used by vCenter to send logs to a logging server.

    Values:
      PROTOCOL_UNSPECIFIED: Unspecified communications protocol. This is the
        default value.
      UDP: UDP
      TCP: TCP
      TLS: TLS
      SSL: SSL
      RELP: RELP
    """
        PROTOCOL_UNSPECIFIED = 0
        UDP = 1
        TCP = 2
        TLS = 3
        SSL = 4
        RELP = 5

    class SourceTypeValueValuesEnum(_messages.Enum):
        """Required. The type of component that produces logs that will be
    forwarded to this logging server.

    Values:
      SOURCE_TYPE_UNSPECIFIED: The default value. This value should never be
        used.
      ESXI: Logs produced by ESXI hosts
      VCSA: Logs produced by vCenter server
    """
        SOURCE_TYPE_UNSPECIFIED = 0
        ESXI = 1
        VCSA = 2
    createTime = _messages.StringField(1)
    etag = _messages.StringField(2)
    hostname = _messages.StringField(3)
    name = _messages.StringField(4)
    port = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    protocol = _messages.EnumField('ProtocolValueValuesEnum', 6)
    sourceType = _messages.EnumField('SourceTypeValueValuesEnum', 7)
    uid = _messages.StringField(8)
    updateTime = _messages.StringField(9)