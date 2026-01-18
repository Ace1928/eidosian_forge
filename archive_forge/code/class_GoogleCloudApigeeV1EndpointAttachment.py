from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1EndpointAttachment(_messages.Message):
    """Apigee endpoint attachment. For more information, see [Southbound
  networking patterns] (https://cloud.google.com/apigee/docs/api-
  platform/architecture/southbound-networking-patterns-endpoints).

  Enums:
    ConnectionStateValueValuesEnum: Output only. State of the endpoint
      attachment connection to the service attachment.
    StateValueValuesEnum: Output only. State of the endpoint attachment.
      Values other than `ACTIVE` mean the resource is not ready to use.

  Fields:
    connectionState: Output only. State of the endpoint attachment connection
      to the service attachment.
    host: Output only. Host that can be used in either the HTTP target
      endpoint directly or as the host in target server.
    location: Required. Location of the endpoint attachment.
    name: Name of the endpoint attachment. Use the following structure in your
      request: `organizations/{org}/endpointAttachments/{endpoint_attachment}`
    serviceAttachment: Format: projects/*/regions/*/serviceAttachments/*
    state: Output only. State of the endpoint attachment. Values other than
      `ACTIVE` mean the resource is not ready to use.
  """

    class ConnectionStateValueValuesEnum(_messages.Enum):
        """Output only. State of the endpoint attachment connection to the
    service attachment.

    Values:
      CONNECTION_STATE_UNSPECIFIED: The connection state has not been set.
      UNAVAILABLE: The connection state is unavailable at this time, possibly
        because the endpoint attachment is currently being provisioned.
      PENDING: The connection is pending acceptance by the PSC producer.
      ACCEPTED: The connection has been accepted by the PSC producer.
      REJECTED: The connection has been rejected by the PSC producer.
      CLOSED: The connection has been closed by the PSC producer and will not
        serve traffic going forward.
      FROZEN: The connection has been frozen by the PSC producer and will not
        serve traffic.
      NEEDS_ATTENTION: The connection has been accepted by the PSC producer,
        but it is not ready to serve the traffic due to producer side issues.
    """
        CONNECTION_STATE_UNSPECIFIED = 0
        UNAVAILABLE = 1
        PENDING = 2
        ACCEPTED = 3
        REJECTED = 4
        CLOSED = 5
        FROZEN = 6
        NEEDS_ATTENTION = 7

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the endpoint attachment. Values other than
    `ACTIVE` mean the resource is not ready to use.

    Values:
      STATE_UNSPECIFIED: Resource is in an unspecified state.
      CREATING: Resource is being created.
      ACTIVE: Resource is provisioned and ready to use.
      DELETING: The resource is being deleted.
      UPDATING: The resource is being updated.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        UPDATING = 4
    connectionState = _messages.EnumField('ConnectionStateValueValuesEnum', 1)
    host = _messages.StringField(2)
    location = _messages.StringField(3)
    name = _messages.StringField(4)
    serviceAttachment = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)