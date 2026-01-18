import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
def _create_status_matcher(self):
    expected = self.expected

    def acceptor_matches(response):
        status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        return status_code == expected
    return acceptor_matches