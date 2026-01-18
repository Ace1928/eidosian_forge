import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
class ECSSecurityGroupAttribute:
    """
    Security group attribute.
    """

    def __init__(self, ip_protocol=None, port_range=None, source_group_id=None, policy=None, nic_type=None):
        self.ip_protocol = ip_protocol
        self.port_range = port_range
        self.source_group_id = source_group_id
        self.policy = policy
        self.nic_type = nic_type

    def __repr__(self):
        return '<ECSSecurityGroupAttribute: ip_protocol=%s ...>' % self.ip_protocol