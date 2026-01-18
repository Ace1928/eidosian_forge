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
class LimitsResource(ks_flask.ResourceBase):
    collection_key = 'limits'
    member_key = 'limit'
    json_home_resource_status = json_home.Status.EXPERIMENTAL
    get_member_from_driver = PROVIDERS.deferred_provider_lookup(api='unified_limit_api', method='get_limit')

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

    def _get_limit(self, limit_id):
        ENFORCER.enforce_call(action='identity:get_limit', build_target=_build_limit_enforcement_target)
        ref = PROVIDERS.unified_limit_api.get_limit(limit_id)
        return self.wrap_member(ref)

    def get(self, limit_id=None):
        if limit_id is not None:
            return self._get_limit(limit_id)
        return self._list_limits()

    def post(self):
        ENFORCER.enforce_call(action='identity:create_limits')
        limits_b = (flask.request.get_json(silent=True, force=True) or {}).get('limits', {})
        validation.lazy_validate(schema.limit_create, limits_b)
        limits = [self._assign_unique_id(self._normalize_dict(limit)) for limit in limits_b]
        refs = PROVIDERS.unified_limit_api.create_limits(limits)
        refs = self.wrap_collection(refs)
        refs.pop('links')
        return (refs, http.client.CREATED)

    def patch(self, limit_id):
        ENFORCER.enforce_call(action='identity:update_limit')
        limit = (flask.request.get_json(silent=True, force=True) or {}).get('limit', {})
        validation.lazy_validate(schema.limit_update, limit)
        self._require_matching_id(limit)
        ref = PROVIDERS.unified_limit_api.update_limit(limit_id, limit)
        return self.wrap_member(ref)

    def delete(self, limit_id):
        ENFORCER.enforce_call(action='identity:delete_limit')
        return (PROVIDERS.unified_limit_api.delete_limit(limit_id), http.client.NO_CONTENT)