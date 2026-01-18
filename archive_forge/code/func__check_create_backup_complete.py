from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import progress
from heat.engine import resource
from heat.engine import rsrc_defn
def _check_create_backup_complete(self, prg):
    backup = self.client().backups.get(prg.backup_id)
    if backup.status == 'creating':
        return False
    if backup.status == 'available':
        return True
    else:
        raise exception.ResourceUnknownStatus(resource_status=backup.status, result=_('Volume backup failed'))