import base64
import hashlib
from libcloud.utils.py3 import b, next, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks, exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.utils.escape import sanitize_object_name
from libcloud.storage.types import ObjectDoesNotExistError, ContainerDoesNotExistError
from libcloud.storage.providers import Provider
def _to_object(self, item, container=None):
    extra = {}
    extra['fileId'] = item['fileId']
    extra['uploadTimestamp'] = item.get('uploadTimestamp', None)
    size = item.get('size', item.get('contentLength', None))
    hash = item.get('contentSha1', None)
    meta_data = item.get('fileInfo', {})
    obj = Object(name=item['fileName'], size=size, hash=hash, extra=extra, meta_data=meta_data, container=container, driver=self)
    return obj