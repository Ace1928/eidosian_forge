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
def _construct_s3_control_endpoint(self, region_name, account):
    self._validate_host_labels(region_name, account)
    if self._endpoint_url:
        endpoint_url_netloc = urlsplit(self._endpoint_url).netloc
        netloc = [account, endpoint_url_netloc]
    else:
        netloc = [account, 's3-control']
        self._add_dualstack(netloc)
        dns_suffix = self._get_dns_suffix(region_name)
        netloc.extend([region_name, dns_suffix])
    return self._construct_netloc(netloc)