import os
import sys
import time
import errno
import base64
import logging
import datetime
import urllib.parse
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from libcloud.utils.py3 import b, httplib, urlparse, urlencode
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError
from libcloud.utils.connection import get_response_object
def _utc_timestamp(datetime_obj):
    """
    Return string of datetime_obj in the UTC Timestamp Format
    """
    return datetime_obj.strftime(UTC_TIMESTAMP_FORMAT)