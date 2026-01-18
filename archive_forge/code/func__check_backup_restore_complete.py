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
def _check_backup_restore_complete(self):
    vol = self.client().volumes.get(self.resource_id)
    if vol.status == 'restoring-backup':
        LOG.debug('Volume %s is being restoring from backup', vol.id)
        return False
    if vol.status != 'available':
        LOG.info('Restore failed: Volume %(vol)s is in %(status)s state.', {'vol': vol.id, 'status': vol.status})
        raise exception.ResourceUnknownStatus(resource_status=vol.status, result=_('Volume backup restore failed'))
    LOG.info('Volume %s backup restore complete', vol.id)
    return True