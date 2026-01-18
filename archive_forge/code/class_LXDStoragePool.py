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
class LXDStoragePool:
    """
    Utility class representing an LXD storage pool
    https://lxd.readthedocs.io/en/latest/storage/
    """

    def __init__(self, name, driver, used_by, config, managed):
        self.name = name
        self.driver = driver
        self.used_by = used_by
        self.config = config
        self.managed = managed