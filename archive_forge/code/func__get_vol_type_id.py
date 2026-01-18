from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _get_vol_type_id(self, volume_type):
    id = self.client_plugin().get_volume_type(volume_type)
    return id