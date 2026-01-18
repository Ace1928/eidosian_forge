from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import reverse_dict
from libcloud.common.base import JsonResponse, PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, MemberCondition
from libcloud.compute.drivers.rackspace import RackspaceConnection
def _iso_to_datetime(self, isodate):
    date_formats = ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z')
    date = None
    for date_format in date_formats:
        try:
            date = datetime.strptime(isodate, date_format)
        except ValueError:
            pass
        if date:
            break
    return date