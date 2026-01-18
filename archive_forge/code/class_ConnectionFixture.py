import uuid
import fixtures
from keystoneauth1.fixture import v2
from keystoneauth1.fixture import v3
import os_service_types
class ConnectionFixture(fixtures.Fixture):
    _suffixes = {'baremetal': '/', 'block-storage': '/{project_id}', 'compute': '/v2.1/', 'container-infrastructure-management': '/v1', 'object-store': '/v1/{project_id}', 'orchestration': '/v1/{project_id}', 'volumev2': '/v2/{project_id}', 'volumev3': '/v3/{project_id}'}

    def __init__(self, suburl=False, project_id=None, *args, **kwargs):
        super(ConnectionFixture, self).__init__(*args, **kwargs)
        self._endpoint_templates = _ENDPOINT_TEMPLATES
        if suburl:
            self.use_suburl()
        self.project_id = project_id or uuid.uuid4().hex.replace('-', '')
        self.build_tokens()

    def use_suburl(self):
        self._endpoint_templates = _SUBURL_TEMPLATES

    def _get_endpoint_templates(self, service_type, alias=None, v2=False):
        templates = {}
        for k, v in self._endpoint_templates.items():
            suffix = self._suffixes.get(alias, self._suffixes.get(service_type, ''))
            if v2:
                suffix = '/v2.0'
            templates[k] = (v + suffix).format(service_type=service_type, project_id=self.project_id)
        return templates

    def _setUp(self):
        pass

    def clear_tokens(self):
        self.v2_token = v2.Token(tenant_id=self.project_id)
        self.v3_token = v3.Token(project_id=self.project_id)

    def build_tokens(self):
        self.clear_tokens()
        for service in _service_type_manager.services:
            service_type = service['service_type']
            if service_type == 'ec2-api':
                continue
            service_name = service['project']
            ets = self._get_endpoint_templates(service_type)
            v3_svc = self.v3_token.add_service(service_type, name=service_name)
            v2_svc = self.v2_token.add_service(service_type, name=service_name)
            v3_svc.add_standard_endpoints(region='RegionOne', **ets)
            if service_type == 'identity':
                ets = self._get_endpoint_templates(service_type, v2=True)
            v2_svc.add_endpoint(region='RegionOne', **ets)
            for alias in service.get('aliases', []):
                ets = self._get_endpoint_templates(service_type, alias=alias)
                v3_svc = self.v3_token.add_service(alias, name=service_name)
                v2_svc = self.v2_token.add_service(alias, name=service_name)
                v3_svc.add_standard_endpoints(region='RegionOne', **ets)
                v2_svc.add_endpoint(region='RegionOne', **ets)

    def _cleanup(self):
        pass