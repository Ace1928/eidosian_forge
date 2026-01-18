import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_utils import timeutils
from urllib import parse as urlparse
from werkzeug import exceptions
from keystone.api._shared import json_home_relations
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.oauth1 import core as oauth1
from keystone.oauth1 import schema
from keystone.oauth1 import validator
from keystone.server import flask as ks_flask
class OSAuth1API(ks_flask.APIBase):
    _name = 'OS-OAUTH1'
    _import_name = __name__
    _api_url_prefix = '/OS-OAUTH1'
    resources = [ConsumerResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=RequestTokenResource, url='/request_token', resource_kwargs={}, rel='request_tokens', resource_relation_func=_build_resource_relation), ks_flask.construct_resource_map(resource=AccessTokenResource, url='/access_token', rel='access_tokens', resource_kwargs={}, resource_relation_func=_build_resource_relation), ks_flask.construct_resource_map(resource=AuthorizeResource, url='/authorize/<string:request_token_id>', resource_kwargs={}, rel='authorize_request_token', resource_relation_func=_build_resource_relation, path_vars={'request_token_id': _build_parameter_relation(parameter_name='request_token_id')})]