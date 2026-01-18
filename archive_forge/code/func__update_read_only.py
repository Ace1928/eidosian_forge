from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import volume_base as vb
from heat.engine import support
from heat.engine import translation
def _update_read_only(self, read_only_flag):
    if read_only_flag is not None:
        self.client().volumes.update_readonly_flag(self.resource_id, read_only_flag)
    return True