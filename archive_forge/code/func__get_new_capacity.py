from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import function
from heat.engine.notification import autoscaling as notification
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.engine import support
from heat.scaling import cooldown
from heat.scaling import scalingutil as sc_util
def _get_new_capacity(self, capacity, adjustment, adjustment_type=sc_util.CFN_EXACT_CAPACITY, min_adjustment_step=None):
    lower = self.properties[self.MIN_SIZE]
    upper = self.properties[self.MAX_SIZE]
    return sc_util.calculate_new_capacity(capacity, adjustment, adjustment_type, min_adjustment_step, lower, upper)