import os
import glance_store as store_api
from oslo_config import cfg
from oslo_log import log as logging
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.types import failure
from glance.common import exception
from glance.i18n import _, _LE
class _CopyImage(task.Task):
    default_provides = 'file_uri'

    def __init__(self, task_id, task_type, image_repo, action_wrapper):
        self.task_id = task_id
        self.task_type = task_type
        self.image_repo = image_repo
        self.image_id = action_wrapper.image_id
        self.action_wrapper = action_wrapper
        super(_CopyImage, self).__init__(name='%s-CopyImage-%s' % (task_type, task_id))
        self.staging_store = store_api.get_store_from_store_identifier('os_glance_staging_store')

    def execute(self):
        with self.action_wrapper as action:
            return self._execute(action)

    def _execute(self, action):
        """Create temp file into store and return path to it

        :param image_id: Glance Image ID
        """
        file_path = '%s/%s' % (getattr(CONF, 'os_glance_staging_store').filesystem_store_datadir, self.image_id)
        if os.path.exists(file_path):
            size_in_staging = os.path.getsize(file_path)
            if action.image_size == size_in_staging:
                return (file_path, 0)
            else:
                LOG.debug('Found partial image data in staging %(fn)s, deleting it to re-stage again', {'fn': file_path})
                try:
                    os.unlink(file_path)
                except OSError as e:
                    LOG.error(_LE('Deletion of staged image data from %(fn)s has failed because [Errno %(en)d]'), {'fn': file_path, 'en': e.errno})
                    raise
        default_store = CONF.glance_store.default_backend
        for loc in action.image_locations:
            if loc['metadata'].get('store') == default_store:
                try:
                    return self._copy_to_staging_store(loc)
                except store_api.exceptions.NotFound:
                    msg = _LE('Image not present in default store, searching in all glance-api specific available stores')
                    LOG.error(msg)
                    break
        available_backends = CONF.enabled_backends
        for loc in action.image_locations:
            image_backend = loc['metadata'].get('store')
            if image_backend in available_backends.keys() and image_backend != default_store:
                try:
                    return self._copy_to_staging_store(loc)
                except store_api.exceptions.NotFound:
                    LOG.error(_LE('Image: %(img_id)s is not present in store %(store)s.'), {'img_id': self.image_id, 'store': image_backend})
                    continue
        raise exception.NotFound(_('Image not found in any configured store'))

    def _copy_to_staging_store(self, loc):
        store_backend = loc['metadata'].get('store')
        image_data, size = store_api.get(loc['url'], store_backend)
        msg = 'Found image, copying it in staging area'
        LOG.debug(msg)
        return self.staging_store.add(self.image_id, image_data, size)[0]

    def revert(self, result, **kwargs):
        if isinstance(result, failure.Failure):
            LOG.error(_LE('Task: %(task_id)s failed to copy image %(image_id)s.'), {'task_id': self.task_id, 'image_id': self.image_id})