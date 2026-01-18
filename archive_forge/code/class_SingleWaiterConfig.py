import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
class SingleWaiterConfig:
    """Represents the waiter configuration for a single waiter.

    A single waiter is considered the configuration for a single
    value associated with a named waiter (i.e TableExists).

    """

    def __init__(self, single_waiter_config):
        self._config = single_waiter_config
        self.description = single_waiter_config.get('description', '')
        self.operation = single_waiter_config['operation']
        self.delay = single_waiter_config['delay']
        self.max_attempts = single_waiter_config['maxAttempts']

    @property
    def acceptors(self):
        acceptors = []
        for acceptor_config in self._config['acceptors']:
            acceptor = AcceptorConfig(acceptor_config)
            acceptors.append(acceptor)
        return acceptors