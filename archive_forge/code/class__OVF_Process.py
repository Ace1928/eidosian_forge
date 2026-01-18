import os
import re
import shutil
import tarfile
import urllib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from taskflow.patterns import linear_flow as lf
from taskflow import task
from glance.i18n import _, _LW
class _OVF_Process(task.Task):
    """
    Extracts the single disk image from an OVA tarball and saves it to the
    Glance image store. It also parses the included OVF file for selected
    metadata which it then saves in the image store as the previously saved
    image's properties.
    """
    default_provides = 'file_path'

    def __init__(self, task_id, task_type, image_repo):
        self.task_id = task_id
        self.task_type = task_type
        self.image_repo = image_repo
        super(_OVF_Process, self).__init__(name='%s-OVF_Process-%s' % (task_type, task_id))

    def _get_extracted_file_path(self, image_id):
        file_path = CONF.task.work_dir
        if CONF.enabled_backends:
            file_path = getattr(CONF, 'os_glance_tasks_store').filesystem_store_datadir
        return os.path.join(file_path, '%s.extracted' % image_id)

    def _get_ova_iter_objects(self, uri):
        """Returns iterable object either for local file or uri

        :param uri: uri (remote or local) to the ova package we want to iterate
        """
        if uri.startswith('file://'):
            uri = uri.split('file://')[-1]
            return open(uri, 'rb')
        return urllib.request.urlopen(uri)

    def execute(self, image_id, file_path):
        """
        :param image_id: Id to use when storing extracted image to Glance
            image store. It is assumed that some other task has already
            created a row in the store with this id.
        :param file_path: Path to the OVA package
        """
        file_abs_path = file_path.split('file://')[-1]
        image = self.image_repo.get(image_id)
        if image.container_format == 'ova':
            if image.context and image.context.is_admin:
                extractor = OVAImageExtractor()
                data_iter = None
                try:
                    data_iter = self._get_ova_iter_objects(file_path)
                    disk, properties = extractor.extract(data_iter)
                    image.extra_properties.update(properties)
                    image.container_format = 'bare'
                    self.image_repo.save(image)
                    dest_path = self._get_extracted_file_path(image_id)
                    with open(dest_path, 'wb') as f:
                        shutil.copyfileobj(disk, f, 4096)
                finally:
                    if data_iter:
                        data_iter.close()
                os.unlink(file_abs_path)
                os.rename(dest_path, file_abs_path)
            else:
                raise RuntimeError(_('OVA extract is limited to admin'))
        return file_path

    def revert(self, image_id, result, **kwargs):
        fs_path = self._get_extracted_file_path(image_id)
        if os.path.exists(fs_path):
            os.path.remove(fs_path)