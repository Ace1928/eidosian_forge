import os
import hmac
import time
import base64
from typing import Dict, Optional
from hashlib import sha1
from datetime import datetime
import libcloud.utils.py3
from libcloud.utils.py3 import b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.aws import (
from libcloud.common.base import RawResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _get_container_path(self, container):
    """
        Return a container path

        :param container: Container instance
        :type  container: :class:`Container`

        :return: A path for this container.
        :rtype: ``str``
        """
    return '/%s' % container.name