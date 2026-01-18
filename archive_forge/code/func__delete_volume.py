from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import progress
from heat.engine import resource
from heat.engine import rsrc_defn
def _delete_volume(self):
    """Call the volume delete API.

        Returns False if further checking of volume status is required,
        True otherwise.
        """
    try:
        cinder = self.client()
        vol = cinder.volumes.get(self.resource_id)
        if vol.status == 'in-use':
            raise exception.Error(_('Volume in use'))
        if vol.status != 'deleting':
            cinder.volumes.delete(self.resource_id)
        return False
    except Exception as ex:
        self.client_plugin().ignore_not_found(ex)
        return True