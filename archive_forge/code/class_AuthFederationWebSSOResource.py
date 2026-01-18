import string
import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import strutils
import urllib
import werkzeug.exceptions
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.api._shared import saml
from keystone.auth import schema as auth_schema
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import utils as k_utils
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.federation import schema as federation_schema
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
class AuthFederationWebSSOResource(_AuthFederationWebSSOBase):

    @classmethod
    def _perform_auth(cls, protocol_id):
        idps = PROVIDERS.federation_api.list_idps()
        remote_id = None
        for idp in idps:
            try:
                remote_id_name = federation_utils.get_remote_id_parameter(idp, protocol_id)
            except exception.FederatedProtocolNotFound:
                continue
            remote_id = flask.request.environ.get(remote_id_name)
            if remote_id:
                break
        if not remote_id:
            msg = 'Missing entity ID from environment'
            tr_msg = _('Missing entity ID from environment')
            LOG.error(msg)
            raise exception.Unauthorized(tr_msg)
        host = _get_sso_origin_host()
        ref = PROVIDERS.federation_api.get_idp_from_remote_id(remote_id)
        identity_provider = ref['idp_id']
        token = authentication.federated_authenticate_for_token(identity_provider=identity_provider, protocol_id=protocol_id)
        return cls._render_template_response(host, token.id)

    @ks_flask.unenforced_api
    def get(self, protocol_id):
        return self._perform_auth(protocol_id)

    @ks_flask.unenforced_api
    def post(self, protocol_id):
        return self._perform_auth(protocol_id)