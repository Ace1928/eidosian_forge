from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import signal_responder
from heat.engine import support
from heat.scaling import scalingutil as sc_util
def _validate_min_adjustment_step(self):
    adjustment_type = self.properties.get(self.ADJUSTMENT_TYPE)
    adjustment_step = self.properties.get(self.MIN_ADJUSTMENT_STEP)
    if adjustment_type != sc_util.PERCENT_CHANGE_IN_CAPACITY and adjustment_step is not None:
        raise exception.ResourcePropertyValueDependency(prop1=self.MIN_ADJUSTMENT_STEP, prop2=self.ADJUSTMENT_TYPE, value=sc_util.PERCENT_CHANGE_IN_CAPACITY)