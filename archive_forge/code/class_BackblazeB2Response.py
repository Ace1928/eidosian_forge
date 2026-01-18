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
class BackblazeB2Response(JsonResponse):

    def success(self):
        return self.status in [httplib.OK, httplib.CREATED, httplib.ACCEPTED]

    def parse_error(self):
        status = int(self.status)
        body = self.parse_body()
        if status == httplib.UNAUTHORIZED:
            raise InvalidCredsError(body['message'])
        return self.body