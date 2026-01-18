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
def _parse_timestamp_with_tzinfo(value, tzinfo):
    """Parse timestamp with pluggable tzinfo options."""
    if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(value, tzinfo())
    else:
        try:
            return datetime.datetime.fromtimestamp(float(value), tzinfo())
        except (TypeError, ValueError):
            pass
    try:
        return dateutil.parser.parse(value, tzinfos={'GMT': tzutc()})
    except (TypeError, ValueError) as e:
        raise ValueError(f'Invalid timestamp "{value}": {e}')