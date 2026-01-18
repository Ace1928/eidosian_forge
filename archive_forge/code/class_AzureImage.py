import os
import time
import base64
import binascii
from libcloud.utils import iso8601
from libcloud.utils.py3 import parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.storage.types import ObjectDoesNotExistError
from libcloud.common.azure_arm import AzureResourceManagementConnection
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import Provider
from libcloud.storage.drivers.azure_blobs import AzureBlobsStorageDriver
class AzureImage(NodeImage):
    """Represents a Marketplace node image that an Azure VM can boot from."""

    def __init__(self, version, sku, offer, publisher, location, driver):
        self.publisher = publisher
        self.offer = offer
        self.sku = sku
        self.version = version
        self.location = location
        urn = '{}:{}:{}:{}'.format(self.publisher, self.offer, self.sku, self.version)
        name = '{} {} {} {}'.format(self.publisher, self.offer, self.sku, self.version)
        super().__init__(urn, name, driver)

    def __repr__(self):
        return '<AzureImage: id=%s, name=%s, location=%s>' % (self.id, self.name, self.location)