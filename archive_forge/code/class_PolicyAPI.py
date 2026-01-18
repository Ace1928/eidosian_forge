import flask_restful
import http.client
from oslo_log import versionutils
from keystone.api._shared import json_home_relations
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone.policy import schema
from keystone.server import flask as ks_flask
class PolicyAPI(ks_flask.APIBase):
    _name = 'policy'
    _import_name = __name__
    resources = [PolicyResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=EndpointPolicyResource, url='/policies/<string:policy_id>/OS-ENDPOINT-POLICY/endpoints', resource_kwargs={}, rel='policy_endpoints', path_vars={'policy_id': json_home.Parameters.POLICY_ID}, resource_relation_func=_resource_rel_func), ks_flask.construct_resource_map(resource=EndpointPolicyAssociations, url='/policies/<string:policy_id>/OS-ENDPOINT-POLICY/endpoints/<string:endpoint_id>', resource_kwargs={}, rel='endpoint_policy_association', path_vars={'policy_id': json_home.Parameters.POLICY_ID, 'endpoint_id': json_home.Parameters.ENDPOINT_ID}, resource_relation_func=_resource_rel_func), ks_flask.construct_resource_map(resource=ServicePolicyAssociations, url='/policies/<string:policy_id>/OS-ENDPOINT-POLICY/services/<string:service_id>', resource_kwargs={}, rel='service_policy_association', path_vars={'policy_id': json_home.Parameters.POLICY_ID, 'service_id': json_home.Parameters.SERVICE_ID}, resource_relation_func=_resource_rel_func), ks_flask.construct_resource_map(resource=ServiceRegionPolicyAssociations, url='/policies/<string:policy_id>/OS-ENDPOINT-POLICY/services/<string:service_id>/regions/<string:region_id>', resource_kwargs={}, rel='region_and_service_policy_association', path_vars={'policy_id': json_home.Parameters.POLICY_ID, 'service_id': json_home.Parameters.SERVICE_ID, 'region_id': json_home.Parameters.REGION_ID}, resource_relation_func=_resource_rel_func)]