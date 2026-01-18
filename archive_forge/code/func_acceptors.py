import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
@property
def acceptors(self):
    acceptors = []
    for acceptor_config in self._config['acceptors']:
        acceptor = AcceptorConfig(acceptor_config)
        acceptors.append(acceptor)
    return acceptors