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
class AzureComputeGalleryImage(NodeImage):
    """Represents a Compute Gallery image that an Azure VM can boot from."""

    def __init__(self, subscription_id, resource_group, gallery, name, driver):
        id = '/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/galleries/%s/images/%s' % (subscription_id, resource_group, gallery, name)
        super().__init__(id, name, driver)

    def __repr__(self):
        return '<AzureComputeGalleryImage: id=%s, name=%s>' % (self.id, self.name)