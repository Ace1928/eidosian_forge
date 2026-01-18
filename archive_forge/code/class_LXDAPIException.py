import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
class LXDAPIException(Exception):
    """
    Basic exception to be thrown when LXD API
    returns with some kind of error
    """

    def __init__(self, message='Unknown Error Occurred', response_dict=None, error_type=''):
        self.message = message
        self.response_dict = response_dict
        self.type = error_type
        super().__init__(message)

    def __str__(self):
        return self.message