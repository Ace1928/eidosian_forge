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
def _validate_create_sources(self):
    exclusive_options, allow_no_size_ops = self._build_exclusive_options()
    size = self.properties.get(self.SIZE)
    if size is None and (len(allow_no_size_ops) != 1 or len(exclusive_options) != 1):
        msg = _('If neither "%(backup_id)s" nor "%(size)s" is provided, one and only one of "%(source_vol)s", "%(snapshot_id)s" must be specified, but currently specified options: %(exclusive_options)s.') % {'backup_id': self.BACKUP_ID, 'size': self.SIZE, 'source_vol': self.SOURCE_VOLID, 'snapshot_id': self.SNAPSHOT_ID, 'exclusive_options': exclusive_options}
        raise exception.StackValidationFailed(message=msg)
    elif size and len(exclusive_options) > 1:
        msg = _('If "%(size)s" is provided, only one of "%(image)s", "%(image_ref)s", "%(source_vol)s", "%(snapshot_id)s" can be specified, but currently specified options: %(exclusive_options)s.') % {'size': self.SIZE, 'image': self.IMAGE, 'image_ref': self.IMAGE_REF, 'source_vol': self.SOURCE_VOLID, 'snapshot_id': self.SNAPSHOT_ID, 'exclusive_options': exclusive_options}
        raise exception.StackValidationFailed(message=msg)