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
class RackspacePTRRecord:

    def __init__(self, id, ip, domain, driver, extra=None):
        self.id = str(id) if id else None
        self.ip = ip
        self.type = RecordType.PTR
        self.domain = domain
        self.driver = driver
        self.extra = extra or {}

    def update(self, domain, extra=None):
        return self.driver.ex_update_ptr_record(record=self, domain=domain, extra=extra)

    def delete(self):
        return self.driver.ex_delete_ptr_record(record=self)

    def __repr__(self):
        return '<{}: ip={}, domain={}, provider={} ...>'.format(self.__class__.__name__, self.ip, self.domain, self.driver.name)