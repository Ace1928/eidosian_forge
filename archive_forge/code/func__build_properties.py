import copy
import itertools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import output
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.heat import resource_group
from heat.engine.resources import signal_responder
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import support
from heat.rpc import api as rpc_api
def _build_properties(self, config_id, action):
    props = {'config_id': config_id, 'action': action, 'input_values': self.properties.get(self.INPUT_VALUES)}
    if self._signal_transport_none():
        props['status'] = SoftwareDeployment.COMPLETE
        props['status_reason'] = _('Not waiting for outputs signal')
    else:
        props['status'] = SoftwareDeployment.IN_PROGRESS
        props['status_reason'] = _('Deploy data available')
    return props