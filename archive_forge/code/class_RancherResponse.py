import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
class RancherResponse(JsonResponse):

    def parse_error(self):
        parsed = super().parse_error()
        if 'fieldName' in parsed:
            return 'Field {} is {}: {} - {}'.format(parsed['fieldName'], parsed['code'], parsed['message'], parsed['detail'])
        else:
            return '{} - {}'.format(parsed['message'], parsed['detail'])

    def success(self):
        return self.status in VALID_RESPONSE_CODES