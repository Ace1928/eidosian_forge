import hmac
import time
import base64
import hashlib
from io import FileIO as file
from libcloud.utils.py3 import b, next, httplib, urlparse, urlquote, urlencode, urlunquote
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError
from libcloud.storage.base import CHUNK_SIZE, Object, Container, StorageDriver
from libcloud.storage.types import (

        Return a generator of objects for the given container.

        :param container: Container instance
        :type container: :class:`Container`

        :param prefix: Filter objects starting with a prefix.
                       Filtering is performed client-side.
        :type  prefix: ``str``

        :param ex_prefix: (Deprecated.) Filter objects starting with a prefix.
                          Filtering is performed client-side.
        :type  ex_prefix: ``str``

        :return: A generator of Object instances.
        :rtype: ``generator`` of :class:`Object`
        