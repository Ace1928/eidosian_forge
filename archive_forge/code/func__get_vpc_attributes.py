import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def _get_vpc_attributes(self, instance):
    vpcs = findall(instance, xpath='VpcAttributes', namespace=self.namespace)
    if len(vpcs) <= 0:
        return None
    return self._get_extra_dict(vpcs[0], RESOURCE_EXTRA_ATTRIBUTES_MAP['vpc_attributes'])