import functools
import re
import wsgiref.util
import http.client
from keystonemiddleware import auth_token
import oslo_i18n
from oslo_log import log
from oslo_serialization import jsonutils
import webob.dec
import webob.exc
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import render_token
from keystone.common import tokenless_auth
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.models import token_model
def _keystone_specific_values(self, token, request_context):
    request_context.token_reference = render_token.render_token_response_from_model(token)
    if token.domain_scoped:
        request_context.is_admin_project = False
        request_context.domain_id = token.domain_id
        request_context.domain_name = token.domain['name']
    if token.oauth_scoped:
        request_context.is_delegated_auth = True
        request_context.oauth_consumer_id = token.access_token['consumer_id']
        request_context.oauth_access_token_id = token.access_token_id
    if token.trust_scoped:
        request_context.is_delegated_auth = True
        request_context.trust_id = token.trust_id
    if token.is_federated:
        request_context.group_ids = []
        for group in token.federated_groups:
            request_context.group_ids.append(group['id'])
    else:
        request_context.group_ids = []