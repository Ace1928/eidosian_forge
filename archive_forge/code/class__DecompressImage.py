import gzip
import os
import shutil
import zipfile
from oslo_log import log as logging
from oslo_utils import encodeutils
from taskflow.patterns import linear_flow as lf
from taskflow import task
class _DecompressImage(task.Task):
    default_provides = 'file_path'

    def __init__(self, context, task_id, task_type, image_repo, image_id):
        self.context = context
        self.task_id = task_id
        self.task_type = task_type
        self.image_repo = image_repo
        self.image_id = image_id
        self.dest_path = ''
        super(_DecompressImage, self).__init__(name='%s-Decompress_Image-%s' % (task_type, task_id))

    def execute(self, file_path, **kwargs):
        src_path = file_path.split('file://')[-1]
        self.dest_path = '%(path)s.uc' % {'path': src_path}
        image = self.image_repo.get(self.image_id)
        if image.container_format == 'compressed':
            return 'file://%s' % src_path
        head = None
        with open(src_path, 'rb') as fd:
            head = fd.read(MAX_HEADER)
        for key, val in MAGIC_NUMBERS.items():
            offset, key = key.split('_')
            offset = int(offset)
            key = '_' + key
            if head.startswith(val, offset):
                globals()[key](src_path, self.dest_path, self.image_id)
                os.replace(self.dest_path, src_path)
        return 'file://%s' % src_path

    def revert(self, result=None, **kwargs):
        if result is not None:
            LOG.debug('Image decompression failed.')
            if os.path.exists(self.dest_path):
                os.remove(self.dest_path)