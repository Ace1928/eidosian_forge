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
@staticmethod
def _id_format(resp):
    """Format the StackId field in the response as an ARN.

        Also, process other IDs into the correct format.
        """
    if 'StackId' in resp:
        identity = identifier.HeatIdentifier(**resp['StackId'])
        resp['StackId'] = identity.arn()
    if 'EventId' in resp:
        identity = identifier.EventIdentifier(**resp['EventId'])
        resp['EventId'] = identity.event_id
    return resp