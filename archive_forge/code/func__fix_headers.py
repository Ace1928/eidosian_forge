import os
import hmac
import base64
import hashlib
import binascii
from datetime import datetime, timedelta
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import fixxpath
from libcloud.utils.files import read_in_chunks
from libcloud.common.azure import AzureConnection, AzureActiveDirectoryConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _fix_headers(self, headers):
    """
        Update common HTTP headers to their equivalent in Azure Storage

        :param headers: The headers dictionary to be updated
        :type headers: ``dict``
        """
    to_fix = ('cache-control', 'content-encoding', 'content-language')
    fixed = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower in to_fix:
            fixed['x-ms-blob-%s' % key_lower] = value
        else:
            fixed[key] = value
    return fixed