import errno
import hashlib
import os.path  # pylint: disable-msg=W0404
import warnings
from typing import Dict, List, Type, Iterator, Optional
from os.path import join as pjoin
import libcloud.utils.files
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.storage.types import ObjectDoesNotExistError
def _filter_listed_container_objects(self, objects, prefix):
    if prefix is not None:
        warnings.warn('Driver %s does not implement native object filtering; falling back to filtering the full object stream.' % self.__class__.__name__)
    for obj in objects:
        if prefix is None or obj.name.startswith(prefix):
            yield obj