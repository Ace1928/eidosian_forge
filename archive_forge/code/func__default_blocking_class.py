from jupyter_client.manager import KernelManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_client.session import Session
from traitlets import DottedObjectName, Instance, default
from .constants import INPROCESS_KEY
@default('blocking_class')
def _default_blocking_class(self):
    from .blocking import BlockingInProcessKernelClient
    return BlockingInProcessKernelClient