import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def _select_endpoint(self, endpoints):
    return endpoints[0]['Address']