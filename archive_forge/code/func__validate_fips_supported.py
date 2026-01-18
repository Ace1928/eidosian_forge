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
def _validate_fips_supported(self, request):
    if not self._use_fips_endpoint:
        return
    if 'fips' in request.context['s3_accesspoint']['region']:
        raise UnsupportedS3AccesspointConfigurationError(msg={'Invalid ARN, FIPS region not allowed in ARN.'})
    if 'outpost_name' in request.context['s3_accesspoint']:
        raise UnsupportedS3AccesspointConfigurationError(msg='Client is configured to use the FIPS psuedo-region "%s", but outpost ARNs do not support FIPS endpoints.' % self._region)
    accesspoint_region = request.context['s3_accesspoint']['region']
    if accesspoint_region != self._region:
        if not self._s3_config.get('use_arn_region', True):
            raise UnsupportedS3AccesspointConfigurationError(msg='Client is configured to use the FIPS psuedo-region for "%s", but the access-point ARN provided is for the "%s" region. For clients using a FIPS psuedo-region calls to access-point ARNs in another region are not allowed.' % (self._region, accesspoint_region))