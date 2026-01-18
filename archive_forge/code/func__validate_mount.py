import copy
import shlex
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine import support
from heat.engine import translation
def _validate_mount(self, mount):
    volume_id = mount.get(self.VOLUME_ID)
    volume_size = mount.get(self.VOLUME_SIZE)
    if volume_id is None and volume_size is None:
        msg = _('One of the properties "%(id)s" or "%(size)s" should be set for the specified mount of container "%(container)s".') % dict(id=self.VOLUME_ID, size=self.VOLUME_SIZE, container=self.name)
        raise exception.StackValidationFailed(message=msg)
    if volume_id and volume_size:
        raise exception.ResourcePropertyConflict('/'.join([self.NETWORKS, self.VOLUME_ID]), '/'.join([self.NETWORKS, self.VOLUME_SIZE]))