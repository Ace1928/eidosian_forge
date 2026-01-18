from os_ken import cfg
import inspect
import platform
import logging
import logging.config
import logging.handlers
import os
import sys
def _get_log_file():
    if CONF.log_file:
        return CONF.log_file
    if CONF.log_dir:
        return os.path.join(CONF.log_dir, os.path.basename(inspect.stack()[-1][1])) + '.log'
    return None