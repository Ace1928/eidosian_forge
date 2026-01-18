from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
class FlowClassifierConstraint(NeutronExtConstraint):
    resource_name = 'flow_classifier'
    extension = 'sfc'