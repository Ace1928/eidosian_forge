import flask
import flask_restful
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone import exception
from keystone.limit import schema
from keystone.server import flask as ks_flask
def _list_limits(self):
    filters = ['service_id', 'region_id', 'resource_name', 'project_id', 'domain_id']
    ENFORCER.enforce_call(action='identity:list_limits', filters=filters)
    hints = self.build_driver_hints(filters)
    filtered_refs = []
    if self.oslo_context.system_scope:
        refs = PROVIDERS.unified_limit_api.list_limits(hints)
        filtered_refs = refs
    elif self.oslo_context.domain_id:
        refs = PROVIDERS.unified_limit_api.list_limits(hints)
        projects = PROVIDERS.resource_api.list_projects_in_domain(self.oslo_context.domain_id)
        project_ids = [project['id'] for project in projects]
        for limit in refs:
            if limit.get('project_id'):
                if limit['project_id'] in project_ids:
                    filtered_refs.append(limit)
            elif limit.get('domain_id'):
                if limit['domain_id'] == self.oslo_context.domain_id:
                    filtered_refs.append(limit)
    elif self.oslo_context.project_id:
        hints.add_filter('project_id', self.oslo_context.project_id)
        refs = PROVIDERS.unified_limit_api.list_limits(hints)
        filtered_refs = refs
    return self.wrap_collection(filtered_refs, hints=hints)