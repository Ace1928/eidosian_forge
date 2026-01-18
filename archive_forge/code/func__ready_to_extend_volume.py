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
def _ready_to_extend_volume(self):
    vol = self.client().volumes.get(self.resource_id)
    expected_status = ('available', 'in-use') if vol.multiattach else ('available',)
    if vol.status in expected_status:
        LOG.debug('Volume %s is ready to extend.', vol.id)
        return True
    return False