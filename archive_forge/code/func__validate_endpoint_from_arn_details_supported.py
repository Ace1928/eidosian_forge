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
def _validate_endpoint_from_arn_details_supported(self, request):
    if 'fips' in request.context['arn_details']['region']:
        raise UnsupportedS3ControlArnError(arn=request.context['arn_details']['original'], msg='Invalid ARN, FIPS region not allowed in ARN.')
    if not self._s3_config.get('use_arn_region', False):
        arn_region = request.context['arn_details']['region']
        if arn_region != self._region:
            error_msg = 'The use_arn_region configuration is disabled but received arn for "%s" when the client is configured to use "%s"' % (arn_region, self._region)
            raise UnsupportedS3ControlConfigurationError(msg=error_msg)
    request_partion = request.context['arn_details']['partition']
    if request_partion != self._partition:
        raise UnsupportedS3ControlConfigurationError(msg='Client is configured for "%s" partition, but arn provided is for "%s" partition. The client and arn partition must be the same.' % (self._partition, request_partion))
    if self._s3_config.get('use_accelerate_endpoint'):
        raise UnsupportedS3ControlConfigurationError(msg='S3 control client does not support accelerate endpoints')
    if 'outpost_name' in request.context['arn_details']:
        self._validate_outpost_redirection_valid(request)