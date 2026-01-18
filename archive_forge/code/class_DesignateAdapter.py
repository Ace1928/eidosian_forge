from keystoneauth1 import adapter
from designateclient import exceptions
from designateclient.v2.blacklists import BlacklistController
from designateclient.v2.limits import LimitController
from designateclient.v2.nameservers import NameServerController
from designateclient.v2.pools import PoolController
from designateclient.v2.quotas import QuotasController
from designateclient.v2.recordsets import RecordSetController
from designateclient.v2.reverse import FloatingIPController
from designateclient.v2.service_statuses import ServiceStatusesController
from designateclient.v2.tlds import TLDController
from designateclient.v2.tsigkeys import TSIGKeysController
from designateclient.v2.zones import ZoneController
from designateclient.v2.zones import ZoneExportsController
from designateclient.v2.zones import ZoneImportsController
from designateclient.v2.zones import ZoneShareController
from designateclient.v2.zones import ZoneTransfersController
from designateclient import version
from oslo_utils import importutils
class DesignateAdapter(adapter.LegacyJsonAdapter):
    """Adapter around LegacyJsonAdapter.

    The user can pass a timeout keyword that will apply only to
    the Designate Client, in order:

    - timeout keyword passed to ``request()``
    - timeout attribute on keystone session
    """

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.pop('timeout', None)
        self.all_projects = kwargs.pop('all_projects', False)
        self.edit_managed = kwargs.pop('edit_managed', False)
        self.hard_delete = kwargs.pop('hard_delete', False)
        self.sudo_project_id = kwargs.pop('sudo_project_id', None)
        super(self.__class__, self).__init__(*args, **kwargs)

    def request(self, *args, **kwargs):
        kwargs.setdefault('raise_exc', False)
        if self.timeout is not None:
            kwargs.setdefault('timeout', self.timeout)
        kwargs.setdefault('headers', {})
        if self.all_projects:
            kwargs['headers'].setdefault('X-Auth-All-Projects', str(self.all_projects))
        if self.edit_managed:
            kwargs['headers'].setdefault('X-Designate-Edit-Managed-Records', str(self.edit_managed))
        if self.hard_delete:
            kwargs['headers'].setdefault('X-Designate-Hard-Delete', str(self.hard_delete))
        if self.sudo_project_id is not None:
            kwargs['headers'].setdefault('X-Auth-Sudo-Project-ID', self.sudo_project_id)
        kwargs['headers'].setdefault('Content-Type', 'application/json')
        if osprofiler_web:
            kwargs['headers'].update(osprofiler_web.get_trace_id_headers())
        response, body = super(self.__class__, self).request(*args, **kwargs)
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = {}
            body = response.text
        if response.status_code == 400:
            raise exceptions.BadRequest(**response_payload)
        elif response.status_code in (401, 403):
            raise exceptions.Forbidden(**response_payload)
        elif response.status_code == 404:
            raise exceptions.NotFound(**response_payload)
        elif response.status_code == 409:
            raise exceptions.Conflict(**response_payload)
        elif response.status_code == 413:
            raise exceptions.OverQuota(**response_payload)
        elif response.status_code >= 500:
            raise exceptions.Unknown(**response_payload)
        return (response, body)