from oslo_serialization import jsonutils
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.cfn import wait_condition_handle as aws_wch
from heat.engine.resources import signal_responder
from heat.engine.resources import wait_condition as wc_base
from heat.engine import support
def _signal_transport_token(self):
    return self.properties.get(self.SIGNAL_TRANSPORT) == self.TOKEN_SIGNAL