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
class ContainerMetadataFetcher:
    TIMEOUT_SECONDS = 2
    RETRY_ATTEMPTS = 3
    SLEEP_TIME = 1
    IP_ADDRESS = '169.254.170.2'
    _ALLOWED_HOSTS = [IP_ADDRESS, '169.254.170.23', 'fd00:ec2::23', 'localhost']

    def __init__(self, session=None, sleep=time.sleep):
        if session is None:
            session = botocore.httpsession.URLLib3Session(timeout=self.TIMEOUT_SECONDS)
        self._session = session
        self._sleep = sleep

    def retrieve_full_uri(self, full_url, headers=None):
        """Retrieve JSON metadata from container metadata.

        :type full_url: str
        :param full_url: The full URL of the metadata service.
            This should include the scheme as well, e.g
            "http://localhost:123/foo"

        """
        self._validate_allowed_url(full_url)
        return self._retrieve_credentials(full_url, headers)

    def _validate_allowed_url(self, full_url):
        parsed = botocore.compat.urlparse(full_url)
        if self._is_loopback_address(parsed.hostname):
            return
        is_whitelisted_host = self._check_if_whitelisted_host(parsed.hostname)
        if not is_whitelisted_host:
            raise ValueError(f"Unsupported host '{parsed.hostname}'.  Can only retrieve metadata from a loopback address or one of these hosts: {', '.join(self._ALLOWED_HOSTS)}")

    def _is_loopback_address(self, hostname):
        try:
            ip = ip_address(hostname)
            return ip.is_loopback
        except ValueError:
            return False

    def _check_if_whitelisted_host(self, host):
        if host in self._ALLOWED_HOSTS:
            return True
        return False

    def retrieve_uri(self, relative_uri):
        """Retrieve JSON metadata from container metadata.

        :type relative_uri: str
        :param relative_uri: A relative URI, e.g "/foo/bar?id=123"

        :return: The parsed JSON response.

        """
        full_url = self.full_url(relative_uri)
        return self._retrieve_credentials(full_url)

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

    def _get_response(self, full_url, headers, timeout):
        try:
            AWSRequest = botocore.awsrequest.AWSRequest
            request = AWSRequest(method='GET', url=full_url, headers=headers)
            response = self._session.send(request.prepare())
            response_text = response.content.decode('utf-8')
            if response.status_code != 200:
                raise MetadataRetrievalError(error_msg=f'Received non 200 response {response.status_code} from container metadata: {response_text}')
            try:
                return json.loads(response_text)
            except ValueError:
                error_msg = 'Unable to parse JSON returned from container metadata services'
                logger.debug('%s:%s', error_msg, response_text)
                raise MetadataRetrievalError(error_msg=error_msg)
        except RETRYABLE_HTTP_ERRORS as e:
            error_msg = f'Received error when attempting to retrieve container metadata: {e}'
            raise MetadataRetrievalError(error_msg=error_msg)

    def full_url(self, relative_uri):
        return f'http://{self.IP_ADDRESS}{relative_uri}'