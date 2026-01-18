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
def _to_storage_pool(self, data):
    """
        Given a dictionary with the storage pool configuration
        it returns a StoragePool object
        :param data: the storage pool configuration
        :return: :class: .StoragePool
        """
    return LXDStoragePool(name=data['name'], driver=data['driver'], used_by=data['used_by'], config=['config'], managed=False)