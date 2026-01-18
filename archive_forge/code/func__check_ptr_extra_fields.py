import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.drivers.openstack import OpenStack_1_1_Response, OpenStack_1_1_Connection
def _check_ptr_extra_fields(device_or_record):
    if not (hasattr(device_or_record, 'extra') and isinstance(device_or_record.extra, dict) and (device_or_record.extra.get('uri') is not None) and (device_or_record.extra.get('service_name') is not None)):
        raise LibcloudError("Can't create PTR Record for %s because it doesn't have a 'uri' and 'service_name' in 'extra'" % device_or_record)