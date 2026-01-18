from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InboundServicesValueListEntryValuesEnum(_messages.Enum):
    """InboundServicesValueListEntryValuesEnum enum type.

    Values:
      INBOUND_SERVICE_UNSPECIFIED: Not specified.
      INBOUND_SERVICE_MAIL: Allows an application to receive mail.
      INBOUND_SERVICE_MAIL_BOUNCE: Allows an application to receive email-
        bound notifications.
      INBOUND_SERVICE_XMPP_ERROR: Allows an application to receive error
        stanzas.
      INBOUND_SERVICE_XMPP_MESSAGE: Allows an application to receive instant
        messages.
      INBOUND_SERVICE_XMPP_SUBSCRIBE: Allows an application to receive user
        subscription POSTs.
      INBOUND_SERVICE_XMPP_PRESENCE: Allows an application to receive a user's
        chat presence.
      INBOUND_SERVICE_CHANNEL_PRESENCE: Registers an application for
        notifications when a client connects or disconnects from a channel.
      INBOUND_SERVICE_WARMUP: Enables warmup requests.
    """
    INBOUND_SERVICE_UNSPECIFIED = 0
    INBOUND_SERVICE_MAIL = 1
    INBOUND_SERVICE_MAIL_BOUNCE = 2
    INBOUND_SERVICE_XMPP_ERROR = 3
    INBOUND_SERVICE_XMPP_MESSAGE = 4
    INBOUND_SERVICE_XMPP_SUBSCRIBE = 5
    INBOUND_SERVICE_XMPP_PRESENCE = 6
    INBOUND_SERVICE_CHANNEL_PRESENCE = 7
    INBOUND_SERVICE_WARMUP = 8