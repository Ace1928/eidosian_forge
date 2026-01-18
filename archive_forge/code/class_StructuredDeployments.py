import collections
import copy
import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.heat import software_config as sc
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import support
class StructuredDeployments(StructuredDeploymentGroup):
    hidden_msg = _('Please use OS::Heat::StructuredDeploymentGroup instead.')
    support_status = support.SupportStatus(status=support.HIDDEN, message=hidden_msg, version='7.0.0', previous_status=support.SupportStatus(status=support.DEPRECATED, version='2014.2'), substitute_class=StructuredDeploymentGroup)