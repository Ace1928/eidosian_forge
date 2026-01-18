import socket
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.api.aws import exception
from heat.api.aws import utils as api_utils
from heat.common import exception as heat_exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
def _get_identity(self, con, stack_name):
    """Generate a stack identifier from the given stack name or ARN.

        In the case of a stack name, the identifier will be looked up in the
        engine over RPC.
        """
    try:
        return dict(identifier.HeatIdentifier.from_arn(stack_name))
    except ValueError:
        return self.rpc_client.identify_stack(con, stack_name)