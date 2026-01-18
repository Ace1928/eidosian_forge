from ironicclient.common.apiclient import exceptions
from ironicclient.common.apiclient.exceptions import *  # noqa
class StateTransitionTimeout(exceptions.ClientException):
    """Timed out while waiting for a requested provision state."""