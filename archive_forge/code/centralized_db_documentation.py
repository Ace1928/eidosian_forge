from contextlib import contextmanager
import os
import stat
import time
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import fileutils
from glance.common import exception
from glance import context
import glance.db
from glance.i18n import _LI, _LW
from glance.image_cache.drivers import base

        Returns cache files in the supplied directory

        :param basepath: Directory to look in for cache files
        