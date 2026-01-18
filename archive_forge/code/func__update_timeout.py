import collections
import copy
import functools
import itertools
import math
from oslo_log import log as logging
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import timeutils
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import function
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import support
from heat.scaling import rolling_update
from heat.scaling import template as scl_template
def _update_timeout(self, batch_cnt, pause_sec):
    total_pause_time = pause_sec * max(batch_cnt - 1, 0)
    if total_pause_time >= self.stack.timeout_secs():
        msg = _('The current update policy will result in stack update timeout.')
        raise ValueError(msg)
    return self.stack.timeout_secs() - total_pause_time