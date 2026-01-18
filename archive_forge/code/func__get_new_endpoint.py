import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
def _get_new_endpoint(original_endpoint, new_endpoint, use_new_scheme=True):
    new_endpoint_components = urlsplit(new_endpoint)
    original_endpoint_components = urlsplit(original_endpoint)
    scheme = original_endpoint_components.scheme
    if use_new_scheme:
        scheme = new_endpoint_components.scheme
    final_endpoint_components = (scheme, new_endpoint_components.netloc, original_endpoint_components.path, original_endpoint_components.query, '')
    final_endpoint = urlunsplit(final_endpoint_components)
    logger.debug(f'Updating URI from {original_endpoint} to {final_endpoint}')
    return final_endpoint