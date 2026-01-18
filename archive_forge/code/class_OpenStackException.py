from libcloud.utils.py3 import ET, httplib
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.types import LibcloudError, MalformedResponseError, KeyPairDoesNotExistError
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.openstack_identity import (
class OpenStackException(ProviderError):
    pass