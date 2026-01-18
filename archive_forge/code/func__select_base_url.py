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
def _select_base_url(self, base_url, config):
    if config is None:
        config = {}
    requires_ipv6 = config.get('ec2_metadata_service_endpoint_mode') == 'ipv6'
    custom_metadata_endpoint = config.get('ec2_metadata_service_endpoint')
    if requires_ipv6 and custom_metadata_endpoint:
        logger.warning('Custom endpoint and IMDS_USE_IPV6 are both set. Using custom endpoint.')
    chosen_base_url = None
    if base_url != METADATA_BASE_URL:
        chosen_base_url = base_url
    elif custom_metadata_endpoint:
        chosen_base_url = custom_metadata_endpoint
    elif requires_ipv6:
        chosen_base_url = METADATA_BASE_URL_IPv6
    else:
        chosen_base_url = METADATA_BASE_URL
    logger.debug('IMDS ENDPOINT: %s' % chosen_base_url)
    if not is_valid_uri(chosen_base_url):
        raise InvalidIMDSEndpointError(endpoint=chosen_base_url)
    return chosen_base_url