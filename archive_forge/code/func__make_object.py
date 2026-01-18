import os
import time
import errno
import shutil
import tempfile
import threading
from hashlib import sha256
from libcloud.utils.py3 import u, relpath
from libcloud.common.base import Connection
from libcloud.utils.files import read_in_chunks, exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _make_object(self, container, object_name):
    """
        Create an object instance

        :param container: Container.
        :type container: :class:`Container`

        :param object_name: Object name.
        :type object_name: ``str``

        :return: Object instance.
        :rtype: :class:`Object`
        """
    full_path = os.path.join(self.base_path, container.name, object_name)
    if os.path.isdir(full_path):
        raise ObjectError(value=None, driver=self, object_name=object_name)
    try:
        stat = os.stat(full_path)
    except Exception:
        raise ObjectDoesNotExistError(value=None, driver=self, object_name=object_name)
    data_hash = self._get_hash_function()
    data_hash.update(u(stat.st_mtime).encode('ascii'))
    data_hash = data_hash.hexdigest()
    extra = {}
    extra['creation_time'] = stat.st_ctime
    extra['access_time'] = stat.st_atime
    extra['modify_time'] = stat.st_mtime
    return Object(name=object_name, size=stat.st_size, extra=extra, driver=self, container=container, hash=data_hash, meta_data=None)