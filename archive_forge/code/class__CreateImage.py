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
class _CreateImage(task.Task):
    default_provides = 'image_id'

    def __init__(self, task_id, task_type, task_repo, image_repo, image_factory):
        self.task_id = task_id
        self.task_type = task_type
        self.task_repo = task_repo
        self.image_repo = image_repo
        self.image_factory = image_factory
        super(_CreateImage, self).__init__(name='%s-CreateImage-%s' % (task_type, task_id))

    def execute(self):
        task = script_utils.get_task(self.task_repo, self.task_id)
        if task is None:
            return
        task_input = script_utils.unpack_task_input(task)
        image = image_import.create_image(self.image_repo, self.image_factory, task_input.get('image_properties'), self.task_id)
        LOG.debug('Task %(task_id)s created image %(image_id)s', {'task_id': task.task_id, 'image_id': image.image_id})
        return image.image_id

    def revert(self, *args, **kwargs):
        result = kwargs.get('result', None)
        if result is not None:
            if kwargs.get('flow_failures', None) is not None:
                image = self.image_repo.get(result)
                LOG.debug('Deleting image whilst reverting.')
                image.delete()
                self.image_repo.remove(image)