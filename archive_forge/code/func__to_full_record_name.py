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
def _to_full_record_name(self, domain, name):
    """
        Build a FQDN from a domain and record name.

        :param domain: Domain name.
        :type domain: ``str``

        :param name: Record name.
        :type name: ``str``
        """
    if name:
        name = '{}.{}'.format(name, domain)
    else:
        name = domain
    return name