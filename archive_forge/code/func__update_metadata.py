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
def _update_metadata(self, headers, meta_data):
    """
        Update the given metadata in the headers

        :param headers: The headers dictionary to be updated
        :type headers: ``dict``

        :param meta_data: Metadata key value pairs
        :type meta_data: ``dict``
        """
    for key, value in list(meta_data.items()):
        key = 'x-ms-meta-%s' % key
        headers[key] = value