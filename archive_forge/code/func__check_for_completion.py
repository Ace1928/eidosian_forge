import functools
from oslo_log import log as logging
from heat.common import environment_format
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils as iso8601utils
from heat.engine import attributes
from heat.engine import environment
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.scaling import lbutils
from heat.scaling import rolling_update
from heat.scaling import template
def _check_for_completion(self, updater):
    while not self.check_update_complete(updater):
        yield