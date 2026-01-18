from datetime import datetime
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.linode import (
class LinodeDNSResponse(LinodeResponse):

    def _make_excp(self, error):
        result = super()._make_excp(error)
        if isinstance(result, LinodeException) and result.code == 5:
            context = self.connection.context
            if context['resource'] == 'zone':
                result = ZoneDoesNotExistError(value='', driver=self.connection.driver, zone_id=context['id'])
            elif context['resource'] == 'record':
                result = RecordDoesNotExistError(value='', driver=self.connection.driver, record_id=context['id'])
        return result