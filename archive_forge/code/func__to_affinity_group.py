import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def _to_affinity_group(self, data):
    affinity_group = CloudStackAffinityGroup(id=data['id'], name=data['name'], group_type=CloudStackAffinityGroupType(data['type']), account=data.get('account', ''), domain=data.get('domain', ''), domainid=data.get('domainid', ''), description=data.get('description', ''), virtualmachine_ids=data.get('virtualmachineIds', ''))
    return affinity_group