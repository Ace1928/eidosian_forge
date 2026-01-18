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
class ImportActionWrapper(object):
    """Wrapper for all the image metadata operations we do during an import.

    This is used to consolidate the changes we make to image metadata during
    an import operation, and can be used with an admin-capable repo to
    enable non-owner controlled modification of that data if desired.

    Use this as a context manager to make multiple changes followed by
    a save of the image in one operation. An _ImportActions object is
    yielded from the context manager, which defines the available operations.

    :param image_repo: The ImageRepo we should use to fetch/save the image
    :param image-id: The ID of the image we should be altering
    """

    def __init__(self, image_repo, image_id, task_id):
        self._image_repo = image_repo
        self._image_id = image_id
        self._task_id = task_id

    def __enter__(self):
        self._image = self._image_repo.get(self._image_id)
        self._image_previous_status = self._image.status
        self._assert_task_lock(self._image)
        return _ImportActions(self._image)

    def __exit__(self, type, value, traceback):
        if type is not None:
            return
        self.assert_task_lock()
        if self._image_previous_status != self._image.status:
            LOG.debug('Image %(image_id)s status changing from %(old_status)s to %(new_status)s', {'image_id': self._image_id, 'old_status': self._image_previous_status, 'new_status': self._image.status})
        self._image_repo.save(self._image, self._image_previous_status)

    @property
    def image_id(self):
        return self._image_id

    def drop_lock_for_task(self):
        """Delete the import lock for our task.

        This is an atomic operation and thus does not require a context
        for the image save. Note that after calling this method, no
        further actions will be allowed on the image.

        :raises: NotFound if the image was not locked by the expected task.
        """
        image = self._image_repo.get(self._image_id)
        self._image_repo.delete_property_atomic(image, 'os_glance_import_task', self._task_id)

    def _assert_task_lock(self, image):
        task_lock = image.extra_properties.get('os_glance_import_task')
        if task_lock != self._task_id:
            LOG.error('Image %(image)s import task %(task)s attempted to take action on image, but other task %(other)s holds the lock; Aborting.', {'image': self._image_id, 'task': self._task_id, 'other': task_lock})
            raise exception.TaskAbortedError()

    def assert_task_lock(self):
        """Assert that we own the task lock on the image.

        :raises: TaskAbortedError if we do not
        """
        image = self._image_repo.get(self._image_id)
        self._assert_task_lock(image)