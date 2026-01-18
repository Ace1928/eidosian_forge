import abc
import logging
from oslo_config import cfg
from oslo_messaging._drivers import base as driver_base
from oslo_messaging import _metrics as metrics
from oslo_messaging import _utils as utils
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import transport as msg_transport
class ClientSendError(exceptions.MessagingException):
    """Raised if we failed to send a message to a target."""

    def __init__(self, target, ex):
        msg = 'Failed to send to target "%s": %s' % (target, ex)
        super(ClientSendError, self).__init__(msg)
        self.target = target
        self.ex = ex