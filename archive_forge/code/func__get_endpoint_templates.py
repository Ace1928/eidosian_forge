import uuid
import fixtures
from keystoneauth1.fixture import v2
from keystoneauth1.fixture import v3
import os_service_types
def _get_endpoint_templates(self, service_type, alias=None, v2=False):
    templates = {}
    for k, v in self._endpoint_templates.items():
        suffix = self._suffixes.get(alias, self._suffixes.get(service_type, ''))
        if v2:
            suffix = '/v2.0'
        templates[k] = (v + suffix).format(service_type=service_type, project_id=self.project_id)
    return templates