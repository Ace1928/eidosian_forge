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
def _extract_user_params(params):
    """Extract a dictionary of user input parameters for the stack.

        In the AWS API parameters, each user parameter appears as two key-value
        pairs with keys of the form below::

          Parameters.member.1.ParameterKey
          Parameters.member.1.ParameterValue
        """
    return api_utils.extract_param_pairs(params, prefix='Parameters', keyname='ParameterKey', valuename='ParameterValue')