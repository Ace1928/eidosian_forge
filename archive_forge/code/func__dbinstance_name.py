from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
def _dbinstance_name(self):
    name = self.properties[self.NAME]
    if name:
        return name
    return self.physical_resource_name()