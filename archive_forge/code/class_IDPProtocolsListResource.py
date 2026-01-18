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
class IDPProtocolsListResource(_IdentityProvidersProtocolsResourceBase):

    def get(self, idp_id):
        """List protocols for an IDP.

        HEAD/GET /OS-FEDERATION/identity_providers/{idp_id}/protocols
        """
        ENFORCER.enforce_call(action='identity:list_protocols')
        protocol_refs = PROVIDERS.federation_api.list_protocols(idp_id)
        protocols = list(protocol_refs)
        collection = self.wrap_collection(protocols)
        for r in collection[self.collection_key]:
            self._add_related_links(r)
        return collection