import json
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from taskflow.patterns import linear_flow as lf
from glance.async_ import utils
from glance.i18n import _LE
class _Introspect(utils.OptionalTask):
    """Taskflow to pull the embedded metadata out of image file"""

    def __init__(self, task_id, task_type, image_repo):
        self.task_id = task_id
        self.task_type = task_type
        self.image_repo = image_repo
        super(_Introspect, self).__init__(name='%s-Introspect-%s' % (task_type, task_id))

    def execute(self, image_id, file_path):
        """Does the actual introspection

        :param image_id: Glance image ID
        :param file_path: Path to the file being introspected
        """
        try:
            stdout, stderr = putils.trycmd('qemu-img', 'info', '--output=json', file_path, prlimit=utils.QEMU_IMG_PROC_LIMITS, log_errors=putils.LOG_ALL_ERRORS)
        except OSError as exc:
            if exc.errno != 2:
                with excutils.save_and_reraise_exception():
                    exc_message = encodeutils.exception_to_unicode(exc)
                    msg = _LE('Failed to execute introspection %(task_id)s: %(exc)s')
                    LOG.error(msg, {'task_id': self.task_id, 'exc': exc_message})
            return
        if stderr:
            raise RuntimeError(stderr)
        metadata = json.loads(stdout)
        new_image = self.image_repo.get(image_id)
        new_image.virtual_size = metadata.get('virtual-size', 0)
        new_image.disk_format = metadata.get('format')
        self.image_repo.save(new_image)
        LOG.debug('%(task_id)s: Introspection successful: %(file)s', {'task_id': self.task_id, 'file': file_path})
        return new_image