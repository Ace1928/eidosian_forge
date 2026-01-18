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
def _ex_delete_old_vhd(self, resource_group, uri):
    try:
        storageAccount, blobContainer, blob = _split_blob_uri(uri)
        keys = self.ex_get_storage_account_keys(resource_group, storageAccount)
        blobdriver = AzureBlobsStorageDriver(storageAccount, keys['key1'], host='{}.blob{}'.format(storageAccount, self.connection.storage_suffix))
        return blobdriver.delete_object(blobdriver.get_object(blobContainer, blob))
    except ObjectDoesNotExistError:
        return True