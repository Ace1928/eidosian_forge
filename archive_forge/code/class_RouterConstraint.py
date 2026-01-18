from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
class RouterConstraint(NeutronConstraint):
    resource_name = 'router'