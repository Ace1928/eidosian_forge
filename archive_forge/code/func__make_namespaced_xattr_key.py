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
def _make_namespaced_xattr_key(key, namespace='user'):
    """
    Create a fully-qualified xattr-key by including the intended namespace.

    Namespacing differs among OSes[1]:

        FreeBSD: user, system
        Linux: user, system, trusted, security
        MacOS X: not needed

    Mac OS X won't break if we include a namespace qualifier, so, for
    simplicity, we always include it.

    --
    [1] http://en.wikipedia.org/wiki/Extended_file_attributes
    """
    namespaced_key = '.'.join([namespace, key])
    return namespaced_key