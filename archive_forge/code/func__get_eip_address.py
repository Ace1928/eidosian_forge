import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def _get_eip_address(self, instance):
    eips = findall(instance, xpath='EipAddress', namespace=self.namespace)
    if len(eips) <= 0:
        return None
    return self._get_extra_dict(eips[0], RESOURCE_EXTRA_ATTRIBUTES_MAP['eip_address_associate'])