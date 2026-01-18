from operator import itemgetter
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import support
Handle updates correctly.

        Implementing handle_update() here is not just an optimization but a
        must, because the default create/delete behavior would delete the
        unchanged part of the extra route set.
        