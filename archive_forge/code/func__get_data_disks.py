import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def _get_data_disks(self, ex_data_disks):
    if isinstance(ex_data_disks, dict):
        data_disks = [ex_data_disks]
    elif isinstance(ex_data_disks, list):
        data_disks = ex_data_disks
    else:
        raise AttributeError('ex_data_disks should be a list of dict')
    mappings = {'size': 'Size', 'category': 'Category', 'snapshot_id': 'SnapshotId', 'disk_name': 'DiskName', 'description': 'Description', 'device': 'Device', 'delete_with_instance': 'DeleteWithInstance'}
    params = {}
    for idx, disk in enumerate(data_disks):
        key_base = 'DataDisk.{}.'.format(idx + 1)
        for attr in mappings.keys():
            if attr in disk:
                if attr == 'delete_with_instance':
                    value = str(disk[attr]).lower()
                else:
                    value = disk[attr]
                params[key_base + mappings[attr]] = value
    return params