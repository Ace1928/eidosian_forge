from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
def _res_get_args(self):
    return [self.resource_id]