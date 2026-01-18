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
def _fetch_metadata_token(self):
    self._assert_enabled()
    url = self._construct_url(self._TOKEN_PATH)
    headers = {'x-aws-ec2-metadata-token-ttl-seconds': self._TOKEN_TTL}
    self._add_user_agent(headers)
    request = botocore.awsrequest.AWSRequest(method='PUT', url=url, headers=headers)
    for i in range(self._num_attempts):
        try:
            response = self._session.send(request.prepare())
            if response.status_code == 200:
                return response.text
            elif response.status_code in (404, 403, 405):
                return None
            elif response.status_code in (400,):
                raise BadIMDSRequestError(request)
        except ReadTimeoutError:
            return None
        except RETRYABLE_HTTP_ERRORS as e:
            logger.debug('Caught retryable HTTP exception while making metadata service request to %s: %s', url, e, exc_info=True)
        except HTTPClientError as e:
            if isinstance(e.kwargs.get('error'), LocationParseError):
                raise InvalidIMDSEndpointError(endpoint=url, error=e)
            else:
                raise
    return None