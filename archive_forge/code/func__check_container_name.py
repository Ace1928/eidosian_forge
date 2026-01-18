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
def _check_container_name(self, container_name):
    """
        Check if the container name is valid

        :param container_name: Container name
        :type container_name: ``str``
        """
    if '/' in container_name or '\\' in container_name:
        raise InvalidContainerNameError(value=None, driver=self, container_name=container_name)