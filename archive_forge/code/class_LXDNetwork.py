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
class LXDNetwork:
    """
    Utility class representing an LXD network
    """

    @classmethod
    def build_from_response(cls, metadata):
        lxd_network = LXDNetwork()
        lxd_network.name = metadata.get('name', None)
        lxd_network.id = metadata.get('name', None)
        lxd_network.description = metadata.get('description', None)
        lxd_network.type = metadata.get('type', None)
        lxd_network.config = metadata.get('config', None)
        lxd_network.status = metadata.get('status', None)
        lxd_network.locations = metadata.get('locations', None)
        lxd_network.managed = metadata.get('managed', None)
        lxd_network.used_by = metadata.get('used_by', None)
        return lxd_network

    def __init__(self):
        self.name = None
        self.id = None
        self.description = None
        self.type = None
        self.config = None
        self.status = None
        self.locations = None
        self.managed = None
        self.used_by = None
        self.extra = {}