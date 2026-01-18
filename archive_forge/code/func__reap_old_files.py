from contextlib import contextmanager
import errno
import os
import stat
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import fileutils
import xattr
from glance.common import exception
from glance.i18n import _, _LI
from glance.image_cache.drivers import base
def _reap_old_files(self, dirpath, entry_type, grace=None):
    now = time.time()
    reaped = 0
    for path in get_all_regular_files(dirpath):
        mtime = os.path.getmtime(path)
        age = now - mtime
        if not grace:
            LOG.debug("No grace period, reaping '%(path)s' immediately", {'path': path})
            delete_cached_file(path)
            reaped += 1
        elif age > grace:
            LOG.debug("Cache entry '%(path)s' exceeds grace period, (%(age)i s > %(grace)i s)", {'path': path, 'age': age, 'grace': grace})
            delete_cached_file(path)
            reaped += 1
    LOG.info(_LI('Reaped %(reaped)s %(entry_type)s cache entries'), {'reaped': reaped, 'entry_type': entry_type})
    return reaped