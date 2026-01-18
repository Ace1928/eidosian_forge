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
class AuthFederationSaml2Resource(_AuthFederationWebSSOBase):

    def get(self):
        raise werkzeug.exceptions.MethodNotAllowed(valid_methods=['POST'])

    @ks_flask.unenforced_api
    def post(self):
        """Exchange a scoped token for a SAML assertion.

        POST /v3/auth/OS-FEDERATION/saml2
        """
        auth = self.request_body_json.get('auth')
        validation.lazy_validate(federation_schema.saml_create, auth)
        response, service_provider = saml.create_base_saml_assertion(auth)
        headers = _build_response_headers(service_provider)
        response = flask.make_response(response.to_string(), http.client.OK)
        for header, value in headers:
            response.headers[header] = value
        return response