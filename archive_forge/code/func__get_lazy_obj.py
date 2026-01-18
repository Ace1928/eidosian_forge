import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
def _get_lazy_obj(self):
    lazy_obj = object.__getattribute__(self, '_lazy_obj')
    if lazy_obj is None:
        lazy_obj = cls(*lazy_init_args, **lazy_init_kwargs)
        object.__setattr__(self, '_lazy_obj', lazy_obj)
    return lazy_obj