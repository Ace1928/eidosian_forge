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
def _backup_restore(self, vol_id, backup_id):
    try:
        self.client().restores.restore(backup_id, vol_id)
    except Exception as ex:
        if self.client_plugin().is_client_exception(ex):
            raise exception.Error(_('Failed to restore volume %(vol)s from backup %(backup)s - %(err)s') % {'vol': vol_id, 'backup': backup_id, 'err': ex})
        else:
            raise
    return True