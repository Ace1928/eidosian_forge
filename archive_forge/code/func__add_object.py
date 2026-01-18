import random
import hashlib
import os.path
from io import FileIO as file
from libcloud.utils.py3 import b
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _add_object(self, container, object_name, size, extra=None):
    container = self.get_container(container.name)
    extra = extra or {}
    meta_data = extra.get('meta_data', {})
    meta_data.update({'cdn_url': 'http://www.test.com/object/%s' % object_name.replace(' ', '_')})
    obj = Object(name=object_name, size=size, extra=extra, hash=None, meta_data=meta_data, container=container, driver=self)
    self._containers[container.name]['objects'][object_name] = obj
    return obj