import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def _get_internet_related_params(self, ex_internet_charge_type, ex_internet_max_bandwidth_in, ex_internet_max_bandwidth_out):
    params = {}
    if ex_internet_charge_type:
        params['InternetChargeType'] = ex_internet_charge_type
        if ex_internet_charge_type.lower() == 'paybytraffic':
            if ex_internet_max_bandwidth_out:
                params['InternetMaxBandwidthOut'] = ex_internet_max_bandwidth_out
            else:
                raise AttributeError('ex_internet_max_bandwidth_out is mandatory for PayByTraffic internet charge type.')
        elif ex_internet_max_bandwidth_out:
            params['InternetMaxBandwidthOut'] = ex_internet_max_bandwidth_out
    if ex_internet_max_bandwidth_in:
        params['InternetMaxBandwidthIn'] = ex_internet_max_bandwidth_in
    return params