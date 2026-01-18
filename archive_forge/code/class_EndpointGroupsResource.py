import flask_restful
import http.client
from keystone.api._shared import json_home_relations
from keystone.api import endpoints as _endpoints_api
from keystone.catalog import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class EndpointGroupsResource(ks_flask.ResourceBase):
    collection_key = 'endpoint_groups'
    member_key = 'endpoint_group'
    api_prefix = '/OS-EP-FILTER'
    json_home_resource_rel_func = _build_resource_relation
    json_home_parameter_rel_func = _build_parameter_relation

    @staticmethod
    def _require_valid_filter(endpoint_group):
        valid_filter_keys = ['service_id', 'region_id', 'interface']
        filters = endpoint_group.get('filters')
        for key in filters.keys():
            if key not in valid_filter_keys:
                raise exception.ValidationError(attribute=' or '.join(valid_filter_keys), target='endpoint_group')

    def _get_endpoint_group(self, endpoint_group_id):
        ENFORCER.enforce_call(action='identity:get_endpoint_group')
        return self.wrap_member(PROVIDERS.catalog_api.get_endpoint_group(endpoint_group_id))

    def _list_endpoint_groups(self):
        filters = 'name'
        ENFORCER.enforce_call(action='identity:list_endpoint_groups', filters=filters)
        hints = self.build_driver_hints(filters)
        refs = PROVIDERS.catalog_api.list_endpoint_groups(hints)
        return self.wrap_collection(refs, hints=hints)

    def get(self, endpoint_group_id=None):
        if endpoint_group_id is not None:
            return self._get_endpoint_group(endpoint_group_id)
        return self._list_endpoint_groups()

    def post(self):
        ENFORCER.enforce_call(action='identity:create_endpoint_group')
        ep_group = self.request_body_json.get('endpoint_group', {})
        validation.lazy_validate(schema.endpoint_group_create, ep_group)
        if not ep_group.get('filters'):
            msg = _('%s field is required and cannot be empty') % 'filters'
            raise exception.ValidationError(message=msg)
        self._require_valid_filter(ep_group)
        ep_group = self._assign_unique_id(ep_group)
        return (self.wrap_member(PROVIDERS.catalog_api.create_endpoint_group(ep_group['id'], ep_group)), http.client.CREATED)

    def patch(self, endpoint_group_id):
        ENFORCER.enforce_call(action='identity:update_endpoint_group')
        ep_group = self.request_body_json.get('endpoint_group', {})
        validation.lazy_validate(schema.endpoint_group_update, ep_group)
        if 'filters' in ep_group:
            self._require_valid_filter(ep_group)
        self._require_matching_id(ep_group)
        return self.wrap_member(PROVIDERS.catalog_api.update_endpoint_group(endpoint_group_id, ep_group))

    def delete(self, endpoint_group_id):
        ENFORCER.enforce_call(action='identity:delete_endpoint_group')
        return (PROVIDERS.catalog_api.delete_endpoint_group(endpoint_group_id), http.client.NO_CONTENT)