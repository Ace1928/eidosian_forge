from oslo_context import context
from oslo_log import log as logging
import webob.exc
from heat.common.i18n import _
from heat.rpc import client as rpc_client
Redirect client to auth server.

        :param env: wsgi request environment
        :param start_response: wsgi response callback
        :returns: HTTPUnauthorized http response
        