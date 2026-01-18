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
def _build_tokenless_auth_context(self, request):
    """Build the authentication context.

        The context is built from the attributes provided in the env,
        such as certificate and scope attributes.
        """
    tokenless_helper = tokenless_auth.TokenlessAuthHelper(request.environ)
    domain_id, project_id, trust_ref, unscoped, system = tokenless_helper.get_scope()
    user_ref = tokenless_helper.get_mapped_user(project_id, domain_id)
    if user_ref['type'] == federation_utils.UserType.EPHEMERAL:
        auth_context = {}
        auth_context['group_ids'] = user_ref['group_ids']
        auth_context[federation_constants.IDENTITY_PROVIDER] = user_ref[federation_constants.IDENTITY_PROVIDER]
        auth_context[federation_constants.PROTOCOL] = user_ref[federation_constants.PROTOCOL]
        if domain_id and project_id:
            msg = _('Scoping to both domain and project is not allowed')
            raise ValueError(msg)
        if domain_id:
            auth_context['domain_id'] = domain_id
        if project_id:
            auth_context['project_id'] = project_id
        auth_context['roles'] = user_ref['roles']
    else:
        token = token_model.TokenModel()
        token.user_id = user_ref['id']
        token.methods = [CONF.tokenless_auth.protocol]
        token.domain_id = domain_id
        token.project_id = project_id
        auth_context = {'user_id': user_ref['id']}
        auth_context['is_delegated_auth'] = False
        if domain_id:
            auth_context['domain_id'] = domain_id
        if project_id:
            auth_context['project_id'] = project_id
        auth_context['roles'] = [role['name'] for role in token.roles]
    return auth_context