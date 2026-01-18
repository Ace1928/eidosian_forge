from heat.common import exception
from heat.common.i18n import _
from heat.engine import properties
from heat.engine import resource
def access_allowed(self, resource_name):
    return resource_name in self.properties[self.ALLOWED_RESOURCES]