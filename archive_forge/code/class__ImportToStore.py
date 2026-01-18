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
class _ImportToStore(task.Task):

    def __init__(self, task_id, task_type, image_repo, uri, backend):
        self.task_id = task_id
        self.task_type = task_type
        self.image_repo = image_repo
        self.uri = uri
        self.backend = backend
        super(_ImportToStore, self).__init__(name='%s-ImportToStore-%s' % (task_type, task_id))

    def execute(self, image_id, file_path=None):
        """Bringing the introspected image to back end store

        :param image_id: Glance Image ID
        :param file_path: path to the image file
        """
        image = self.image_repo.get(image_id)
        image.status = 'saving'
        self.image_repo.save(image)
        try:
            image_import.set_image_data(image, file_path or self.uri, self.task_id, backend=self.backend)
        except IOError as e:
            msg = _('Uploading the image failed due to: %(exc)s') % {'exc': encodeutils.exception_to_unicode(e)}
            LOG.error(msg)
            raise exception.UploadException(message=msg)
        self.image_repo.save(image)