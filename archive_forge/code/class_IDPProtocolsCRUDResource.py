import flask
import flask_restful
import http.client
from oslo_serialization import jsonutils
from oslo_log import log
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import schema
from keystone.federation import utils
from keystone.server import flask as ks_flask
class IDPProtocolsCRUDResource(_IdentityProvidersProtocolsResourceBase):

    def get(self, idp_id, protocol_id):
        """Get protocols for an IDP.

        HEAD/GET /OS-FEDERATION/identity_providers/
                 {idp_id}/protocols/{protocol_id}
        """
        ENFORCER.enforce_call(action='identity:get_protocol')
        ref = PROVIDERS.federation_api.get_protocol(idp_id, protocol_id)
        return self.wrap_member(ref)

    def put(self, idp_id, protocol_id):
        """Create protocol for an IDP.

        PUT /OS-Federation/identity_providers/{idp_id}/protocols/{protocol_id}
        """
        ENFORCER.enforce_call(action='identity:create_protocol')
        protocol = self.request_body_json.get('protocol', {})
        validation.lazy_validate(schema.protocol_create, protocol)
        protocol = self._normalize_dict(protocol)
        ref = PROVIDERS.federation_api.create_protocol(idp_id, protocol_id, protocol)
        return (self.wrap_member(ref), http.client.CREATED)

    def patch(self, idp_id, protocol_id):
        """Update protocol for an IDP.

        PATCH /OS-FEDERATION/identity_providers/
              {idp_id}/protocols/{protocol_id}
        """
        ENFORCER.enforce_call(action='identity:update_protocol')
        protocol = self.request_body_json.get('protocol', {})
        validation.lazy_validate(schema.protocol_update, protocol)
        ref = PROVIDERS.federation_api.update_protocol(idp_id, protocol_id, protocol)
        return self.wrap_member(ref)

    def delete(self, idp_id, protocol_id):
        """Delete a protocol from an IDP.

        DELETE /OS-FEDERATION/identity_providers/
               {idp_id}/protocols/{protocol_id}
        """
        ENFORCER.enforce_call(action='identity:delete_protocol')
        PROVIDERS.federation_api.delete_protocol(idp_id, protocol_id)
        return (None, http.client.NO_CONTENT)