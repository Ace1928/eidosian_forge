import os
from oslo_concurrency import processutils as putils
from oslo_config import cfg
from oslo_log import log as logging
from taskflow.patterns import linear_flow as lf
from taskflow import task
from glance.i18n import _, _LW
class _Convert(task.Task):
    conversion_missing_warned = False

    def __init__(self, task_id, task_type, image_repo):
        self.task_id = task_id
        self.task_type = task_type
        self.image_repo = image_repo
        super(_Convert, self).__init__(name='%s-Convert-%s' % (task_type, task_id))

    def execute(self, image_id, file_path):
        abs_file_path = file_path.split('file://')[-1]
        conversion_format = CONF.taskflow_executor.conversion_format
        if conversion_format is None:
            if not _Convert.conversion_missing_warned:
                msg = _LW('The conversion format is None, please add a value for it in the config file for this task to work: %s')
                LOG.warning(msg, self.task_id)
                _Convert.conversion_missing_warned = True
            return
        image_obj = self.image_repo.get(image_id)
        src_format = image_obj.disk_format
        data_dir = CONF.task.work_dir
        if CONF.enabled_backends:
            data_dir = getattr(CONF, 'os_glance_tasks_store').filesystem_store_datadir
        dest_path = os.path.join(data_dir, '%s.converted' % image_id)
        stdout, stderr = putils.trycmd('qemu-img', 'convert', '-f', src_format, '-O', conversion_format, file_path, dest_path, log_errors=putils.LOG_ALL_ERRORS)
        if stderr:
            raise RuntimeError(stderr)
        os.unlink(abs_file_path)
        os.rename(dest_path, abs_file_path)
        return file_path

    def revert(self, image_id, result=None, **kwargs):
        if result is None:
            return
        fs_path = result.split('file://')[-1]
        if os.path.exists(fs_path):
            os.remove(fs_path)