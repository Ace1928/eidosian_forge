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
def _get_gce_metadata(path='', retry_failed: Optional[bool]=None):
    try:
        url = 'http://metadata/computeMetadata/v1/' + path.lstrip('/')
        headers = {'Metadata-Flavor': 'Google'}
        response = get_response_object(url, headers=headers, retry_failed=retry_failed)
        return (response.status, '', response.body)
    except Exception as e:
        return (-1, str(e), None)