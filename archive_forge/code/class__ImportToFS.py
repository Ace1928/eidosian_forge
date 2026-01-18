import json
import os
import glance_store as store_api
from glance_store import backend
from oslo_concurrency import processutils as putils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from stevedore import named
from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import task
from taskflow.types import failure
from glance.async_ import utils
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.i18n import _, _LE, _LI
class _ImportToFS(task.Task):
    default_provides = 'file_path'

    def __init__(self, task_id, task_type, task_repo, uri):
        self.task_id = task_id
        self.task_type = task_type
        self.task_repo = task_repo
        self.uri = uri
        super(_ImportToFS, self).__init__(name='%s-ImportToFS-%s' % (task_type, task_id))
        if CONF.enabled_backends:
            self.store = store_api.get_store_from_store_identifier('os_glance_tasks_store')
        else:
            if CONF.task.work_dir is None:
                msg = _('%(task_id)s of %(task_type)s not configured properly. Missing work dir: %(work_dir)s') % {'task_id': self.task_id, 'task_type': self.task_type, 'work_dir': CONF.task.work_dir}
                raise exception.BadTaskConfiguration(msg)
            self.store = self._build_store()

    def _build_store(self):
        conf = cfg.ConfigOpts()
        backend.register_opts(conf)
        conf.set_override('filesystem_store_datadir', CONF.task.work_dir, group='glance_store')
        store = backend._load_store(conf, 'file')
        if store is None:
            msg = _('%(task_id)s of %(task_type)s not configured properly. Could not load the filesystem store') % {'task_id': self.task_id, 'task_type': self.task_type}
            raise exception.BadTaskConfiguration(msg)
        store.configure()
        return store

    def execute(self, image_id):
        """Create temp file into store and return path to it

        :param image_id: Glance Image ID
        """
        data = script_utils.get_image_data_iter(self.uri)
        path = self.store.add(image_id, data, 0, context=None)[0]
        try:
            stdout, stderr = putils.trycmd('qemu-img', 'info', '--output=json', path, prlimit=utils.QEMU_IMG_PROC_LIMITS, log_errors=putils.LOG_ALL_ERRORS)
        except OSError as exc:
            with excutils.save_and_reraise_exception():
                exc_message = encodeutils.exception_to_unicode(exc)
                msg = _LE('Failed to execute security checks on the image %(task_id)s: %(exc)s')
                LOG.error(msg, {'task_id': self.task_id, 'exc': exc_message})
        metadata = json.loads(stdout)
        backing_file = metadata.get('backing-filename')
        if backing_file is not None:
            msg = _('File %(path)s has invalid backing file %(bfile)s, aborting.') % {'path': path, 'bfile': backing_file}
            raise RuntimeError(msg)
        return path

    def revert(self, image_id, result, **kwargs):
        if isinstance(result, failure.Failure):
            LOG.exception(_LE('Task: %(task_id)s failed to import image %(image_id)s to the filesystem.'), {'task_id': self.task_id, 'image_id': image_id})
            return
        if os.path.exists(result.split('file://')[-1]):
            if CONF.enabled_backends:
                store_api.delete(result, 'os_glance_tasks_store')
            else:
                store_api.delete_from_backend(result)