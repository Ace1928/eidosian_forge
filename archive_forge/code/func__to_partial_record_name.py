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
def _to_partial_record_name(self, domain, name):
    """
        Remove domain portion from the record name.

        :param domain: Domain name.
        :type domain: ``str``

        :param name: Full record name (fqdn).
        :type name: ``str``
        """
    if name == domain:
        return None
    name = name.replace('.%s' % domain, '')
    return name