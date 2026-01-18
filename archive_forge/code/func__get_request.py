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
def _get_request(self, url_path, retry_func, token=None):
    """Make a get request to the Instance Metadata Service.

        :type url_path: str
        :param url_path: The path component of the URL to make a get request.
            This arg is appended to the base_url that was provided in the
            initializer.

        :type retry_func: callable
        :param retry_func: A function that takes the response as an argument
             and determines if it needs to retry. By default empty and non
             200 OK responses are retried.

        :type token: str
        :param token: Metadata token to send along with GET requests to IMDS.
        """
    self._assert_enabled()
    if not token:
        self._assert_v1_enabled()
    if retry_func is None:
        retry_func = self._default_retry
    url = self._construct_url(url_path)
    headers = {}
    if token is not None:
        headers['x-aws-ec2-metadata-token'] = token
    self._add_user_agent(headers)
    for i in range(self._num_attempts):
        try:
            request = botocore.awsrequest.AWSRequest(method='GET', url=url, headers=headers)
            response = self._session.send(request.prepare())
            if not retry_func(response):
                return response
        except RETRYABLE_HTTP_ERRORS as e:
            logger.debug('Caught retryable HTTP exception while making metadata service request to %s: %s', url, e, exc_info=True)
    raise self._RETRIES_EXCEEDED_ERROR_CLS()