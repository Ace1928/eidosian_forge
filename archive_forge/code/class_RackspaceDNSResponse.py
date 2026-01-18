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
class RackspaceDNSResponse(OpenStack_1_1_Response):
    """
    Rackspace DNS Response class.
    """

    def parse_error(self):
        status = int(self.status)
        context = self.connection.context
        body = self.parse_body()
        if status == httplib.NOT_FOUND:
            if context['resource'] == 'zone':
                raise ZoneDoesNotExistError(value='', driver=self, zone_id=context['id'])
            elif context['resource'] == 'record':
                raise RecordDoesNotExistError(value='', driver=self, record_id=context['id'])
        if body:
            if 'code' and 'message' in body:
                err = '{} - {} ({})'.format(body['code'], body['message'], body['details'])
                return err
            elif 'validationErrors' in body:
                errors = [m for m in body['validationErrors']['messages']]
                err = 'Validation errors: %s' % ', '.join(errors)
                return err
        raise LibcloudError('Unexpected status code: %s' % status)