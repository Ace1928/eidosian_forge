from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine import attributes
from heat.engine import properties
from heat.engine.resources.openstack.neutron import net
from heat.engine.resources.openstack.neutron import neutron as nr
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class SomeNeutronResource(nr.NeutronResource):
    properties_schema = {}

    @classmethod
    def is_service_available(cls, context):
        return (True, None)