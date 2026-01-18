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
class InstanceMetadataFetcher(IMDSFetcher):
    _URL_PATH = 'latest/meta-data/iam/security-credentials/'
    _REQUIRED_CREDENTIAL_FIELDS = ['AccessKeyId', 'SecretAccessKey', 'Token', 'Expiration']

    def retrieve_iam_role_credentials(self):
        try:
            token = self._fetch_metadata_token()
            role_name = self._get_iam_role(token)
            credentials = self._get_credentials(role_name, token)
            if self._contains_all_credential_fields(credentials):
                credentials = {'role_name': role_name, 'access_key': credentials['AccessKeyId'], 'secret_key': credentials['SecretAccessKey'], 'token': credentials['Token'], 'expiry_time': credentials['Expiration']}
                self._evaluate_expiration(credentials)
                return credentials
            else:
                if 'Code' in credentials and 'Message' in credentials:
                    logger.debug('Error response received when retrievingcredentials: %s.', credentials)
                return {}
        except self._RETRIES_EXCEEDED_ERROR_CLS:
            logger.debug('Max number of attempts exceeded (%s) when attempting to retrieve data from metadata service.', self._num_attempts)
        except BadIMDSRequestError as e:
            logger.debug('Bad IMDS request: %s', e.request)
        return {}

    def _get_iam_role(self, token=None):
        return self._get_request(url_path=self._URL_PATH, retry_func=self._needs_retry_for_role_name, token=token).text

    def _get_credentials(self, role_name, token=None):
        r = self._get_request(url_path=self._URL_PATH + role_name, retry_func=self._needs_retry_for_credentials, token=token)
        return json.loads(r.text)

    def _is_invalid_json(self, response):
        try:
            json.loads(response.text)
            return False
        except ValueError:
            self._log_imds_response(response, 'invalid json')
            return True

    def _needs_retry_for_role_name(self, response):
        return self._is_non_ok_response(response) or self._is_empty(response)

    def _needs_retry_for_credentials(self, response):
        return self._is_non_ok_response(response) or self._is_empty(response) or self._is_invalid_json(response)

    def _contains_all_credential_fields(self, credentials):
        for field in self._REQUIRED_CREDENTIAL_FIELDS:
            if field not in credentials:
                logger.debug('Retrieved credentials is missing required field: %s', field)
                return False
        return True

    def _evaluate_expiration(self, credentials):
        expiration = credentials.get('expiry_time')
        if expiration is None:
            return
        try:
            expiration = datetime.datetime.strptime(expiration, '%Y-%m-%dT%H:%M:%SZ')
            refresh_interval = self._config.get('ec2_credential_refresh_window', 60 * 10)
            jitter = random.randint(120, 600)
            refresh_interval_with_jitter = refresh_interval + jitter
            current_time = datetime.datetime.utcnow()
            refresh_offset = datetime.timedelta(seconds=refresh_interval_with_jitter)
            extension_time = expiration - refresh_offset
            if current_time >= extension_time:
                new_time = current_time + refresh_offset
                credentials['expiry_time'] = new_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                logger.info(f'Attempting credential expiration extension due to a credential service availability issue. A refresh of these credentials will be attempted again within the next {refresh_interval_with_jitter / 60:.0f} minutes.')
        except ValueError:
            logger.debug(f'Unable to parse expiry_time in {credentials['expiry_time']}')