import configparser
import logging
import logging.config
import logging.handlers
import os
import platform
import sys
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_utils import units
from oslo_log._i18n import _
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
def _fix_eventlet_logging():
    """Properly setup logging with eventlet on native threads.

    Workaround for: https://github.com/eventlet/eventlet/issues/432
    """
    if eventletutils.is_monkey_patched('thread'):
        import eventlet.green.threading
        from oslo_log import pipe_mutex
        logging.threading = eventlet.green.threading
        logging._lock = logging.threading.RLock()
        logging.Handler.createLock = pipe_mutex.pipe_createLock