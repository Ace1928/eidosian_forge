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
def _retrieve_credentials(self, full_url, extra_headers=None):
    headers = {'Accept': 'application/json'}
    if extra_headers is not None:
        headers.update(extra_headers)
    attempts = 0
    while True:
        try:
            return self._get_response(full_url, headers, self.TIMEOUT_SECONDS)
        except MetadataRetrievalError as e:
            logger.debug('Received error when attempting to retrieve container metadata: %s', e, exc_info=True)
            self._sleep(self.SLEEP_TIME)
            attempts += 1
            if attempts >= self.RETRY_ATTEMPTS:
                raise