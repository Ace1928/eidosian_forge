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
@staticmethod
def _add_related_links(ref):
    """Add new entries to the 'links' subdictionary in the response.

        Adds 'identity_provider' key with URL pointing to related identity
        provider as a value.

        :param ref: response dictionary

        """
    ref.setdefault('links', {})
    ref['links']['identity_provider'] = ks_flask.base_url(path=ref['idp_id'])