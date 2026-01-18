import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
def _verify_supported_version(self, version):
    if version != self.SUPPORTED_VERSION:
        raise WaiterConfigError(error_msg='Unsupported waiter version, supported version must be: %s, but version of waiter config is: %s' % (self.SUPPORTED_VERSION, version))