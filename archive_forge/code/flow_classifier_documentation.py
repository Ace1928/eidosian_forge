from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
"Heat Template Resource for networking-sfc flow-classifier.

    This resource used to select the traffic that can access the service chain.
    Traffic that matches any flow classifier will be directed to the first
    port in the chain.
    