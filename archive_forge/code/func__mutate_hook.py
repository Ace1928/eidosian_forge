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
def _mutate_hook(conf, fresh):
    """Reconfigures oslo.log according to the mutated options."""
    if (None, 'debug') in fresh:
        _refresh_root_level(conf.debug)
    if (None, 'log-config-append') in fresh:
        _load_log_config.old_time = 0
    if conf.log_config_append:
        _load_log_config(conf.log_config_append)