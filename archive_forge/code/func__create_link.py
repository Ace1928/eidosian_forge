import contextlib
import errno
import io
import os
import shutil
import cachetools
import fasteners
from oslo_serialization import jsonutils
from oslo_utils import fileutils
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.utils import misc
def _create_link(self, src_path, dest_path, transaction):
    with _storagefailure_wrapper():
        try:
            os.symlink(src_path, dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise