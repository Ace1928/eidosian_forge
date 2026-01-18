import atexit
import os
import threading
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import osprofiler.initializer
from glance.api import common
import glance.async_
from glance.common import config
from glance.common import store_utils
from glance import housekeeping
from glance.i18n import _, _LW
from glance import notifier
from glance import sqlite_migration
def _setup_os_profiler():
    notifier.set_defaults()
    if CONF.profiler.enabled:
        osprofiler.initializer.init_from_conf(conf=CONF, context={}, project='glance', service='api', host=CONF.bind_host)