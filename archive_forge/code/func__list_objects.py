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
def _list_objects(self, tree, object_type=None):
    listing = tree.find(self._emc_tag('DirectoryList'))
    entries = []
    for entry in listing.findall(self._emc_tag('DirectoryEntry')):
        file_type = entry.find(self._emc_tag('FileType')).text
        if object_type is not None and object_type != file_type:
            continue
        entries.append({'id': entry.find(self._emc_tag('ObjectID')).text, 'type': file_type, 'name': entry.find(self._emc_tag('Filename')).text})
    return entries