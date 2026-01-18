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
def _extend_volume(self, new_size):
    try:
        self.client().volumes.extend(self.resource_id, new_size)
    except Exception as ex:
        if self.client_plugin().is_client_exception(ex):
            raise exception.Error(_('Failed to extend volume %(vol)s - %(err)s') % {'vol': self.resource_id, 'err': str(ex)})
        else:
            raise
    return True