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
class InstanceMetadataRegionFetcher(IMDSFetcher):
    _URL_PATH = 'latest/meta-data/placement/availability-zone/'

    def retrieve_region(self):
        """Get the current region from the instance metadata service.
        :rvalue: str
        :returns: The region the current instance is running in or None
            if the instance metadata service cannot be contacted or does not
            give a valid response.
        :rtype: None or str
        :returns: Returns the region as a string if it is configured to use
            IMDS as a region source. Otherwise returns ``None``. It will also
            return ``None`` if it fails to get the region from IMDS due to
            exhausting its retries or not being able to connect.
        """
        try:
            region = self._get_region()
            return region
        except self._RETRIES_EXCEEDED_ERROR_CLS:
            logger.debug('Max number of attempts exceeded (%s) when attempting to retrieve data from metadata service.', self._num_attempts)
        return None

    def _get_region(self):
        token = self._fetch_metadata_token()
        response = self._get_request(url_path=self._URL_PATH, retry_func=self._default_retry, token=token)
        availability_zone = response.text
        region = availability_zone[:-1]
        return region