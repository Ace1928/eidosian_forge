from libcloud.utils.py3 import httplib
from libcloud.common.ovh import API_ROOT, OvhConnection
from libcloud.compute.base import (
from libcloud.compute.types import Provider, StorageVolumeState, VolumeSnapshotState
from libcloud.compute.drivers.openstack import OpenStackKeyPair, OpenStackNodeDriver
def _get_project_action(self, suffix):
    base_url = '{}/cloud/project/{}/'.format(API_ROOT, self.project_id)
    return base_url + suffix