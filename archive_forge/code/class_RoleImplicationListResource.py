import flask
import flask_restful
import http.client
from keystone.api._shared import implied_roles as shared
from keystone.assignment import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone.server import flask as ks_flask
class RoleImplicationListResource(flask_restful.Resource):

    def get(self, prior_role_id):
        """List Implied Roles.

        GET/HEAD /v3/roles/{prior_role_id}/implies
        """
        ENFORCER.enforce_call(action='identity:list_implied_roles', build_target=_build_enforcement_target_ref)
        ref = PROVIDERS.role_api.list_implied_roles(prior_role_id)
        implied_ids = [r['implied_role_id'] for r in ref]
        response_json = shared.role_inference_response(prior_role_id)
        response_json['role_inference']['implies'] = []
        for implied_id in implied_ids:
            implied_role = PROVIDERS.role_api.get_role(implied_id)
            response_json['role_inference']['implies'].append(shared.build_implied_role_response_data(implied_role))
        response_json['links'] = {'self': ks_flask.base_url(path='/roles/%s/implies' % prior_role_id)}
        return response_json