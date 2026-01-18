import copy
import functools
import json
import os
import urllib.request
import glance_store as store_api
from glance_store import backend
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_utils import units
import taskflow
from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import task
from glance.api import common as api_common
import glance.async_.flows._internal_plugins as internal_plugins
import glance.async_.flows.plugins as import_plugins
from glance.async_ import utils
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import store_utils
from glance.i18n import _, _LE, _LI
from glance.quota import keystone as ks_quota
class _ImageLock(task.Task):

    def __init__(self, task_id, task_type, action_wrapper):
        self.task_id = task_id
        self.task_type = task_type
        self.action_wrapper = action_wrapper
        super(_ImageLock, self).__init__(name='%s-ImageLock-%s' % (task_type, task_id))

    def execute(self):
        self.action_wrapper.assert_task_lock()
        LOG.debug('Image %(image)s import task %(task)s lock confirmed', {'image': self.action_wrapper.image_id, 'task': self.task_id})

    def revert(self, result, **kwargs):
        """Drop our claim on the image.

        If we have failed, we need to drop our import_task lock on the image
        so that something else can have a try. Note that we may have been
        preempted so we should only drop *our* lock.
        """
        try:
            self.action_wrapper.drop_lock_for_task()
        except exception.NotFound:
            LOG.warning('Image %(image)s import task %(task)s lost its lock during execution!', {'image': self.action_wrapper.image_id, 'task': self.task_id})
        else:
            LOG.debug('Image %(image)s import task %(task)s dropped its lock after failure', {'image': self.action_wrapper.image_id, 'task': self.task_id})