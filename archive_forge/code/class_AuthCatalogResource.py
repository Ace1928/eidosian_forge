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
class AuthCatalogResource(_AuthFederationWebSSOBase):

    def get(self):
        """Get service catalog for token.

        GET/HEAD /v3/auth/catalog
        """
        ENFORCER.enforce_call(action='identity:get_auth_catalog')
        user_id = self.auth_context.get('user_id')
        project_id = self.auth_context.get('project_id')
        if not project_id:
            raise exception.Forbidden(_('A project-scoped token is required to produce a service catalog.'))
        return {'catalog': PROVIDERS.catalog_api.get_v3_catalog(user_id, project_id), 'links': {'self': ks_flask.base_url(path='auth/catalog')}}