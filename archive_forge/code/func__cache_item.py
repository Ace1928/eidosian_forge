import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def _cache_item(self, value):
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    else:
        return value