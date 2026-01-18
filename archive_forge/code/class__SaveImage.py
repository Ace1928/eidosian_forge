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
class _SaveImage(task.Task):

    def __init__(self, task_id, task_type, image_repo):
        self.task_id = task_id
        self.task_type = task_type
        self.image_repo = image_repo
        super(_SaveImage, self).__init__(name='%s-SaveImage-%s' % (task_type, task_id))

    def execute(self, image_id):
        """Transition image status to active

        :param image_id: Glance Image ID
        """
        new_image = self.image_repo.get(image_id)
        if new_image.status == 'saving':
            new_image.status = 'active'
        self.image_repo.save(new_image)