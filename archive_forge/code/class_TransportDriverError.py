import abc
import threading
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_messaging import exceptions
class TransportDriverError(exceptions.MessagingException):
    """Base class for transport driver specific exceptions."""