import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _get_request_audit_info(context, user_id=None):
    """Collect audit information about the request used for CADF.

    :param context: Request context
    :param user_id: Optional user ID, alternatively collected from context
    :returns: Auditing data about the request
    :rtype: :class:`pycadf.Resource`
    """
    remote_addr = None
    http_user_agent = None
    project_id = None
    domain_id = None
    if context and 'environment' in context and context['environment']:
        environment = context['environment']
        remote_addr = environment.get('REMOTE_ADDR')
        http_user_agent = environment.get('HTTP_USER_AGENT')
        if not user_id:
            user_id = environment.get('KEYSTONE_AUTH_CONTEXT', {}).get('user_id')
        project_id = environment.get('KEYSTONE_AUTH_CONTEXT', {}).get('project_id')
        domain_id = environment.get('KEYSTONE_AUTH_CONTEXT', {}).get('domain_id')
    host = pycadf.host.Host(address=remote_addr, agent=http_user_agent)
    initiator = resource.Resource(typeURI=taxonomy.ACCOUNT_USER, host=host)
    if user_id:
        initiator.user_id = user_id
        initiator.id = utils.resource_uuid(user_id)
        initiator = _add_username_to_initiator(initiator)
    if project_id:
        initiator.project_id = project_id
    if domain_id:
        initiator.domain_id = domain_id
    return initiator