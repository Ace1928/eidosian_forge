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
class LogConfigError(Exception):
    message = _('Error loading logging config %(log_config)s: %(err_msg)s')

    def __init__(self, log_config, err_msg):
        self.log_config = log_config
        self.err_msg = err_msg

    def __str__(self):
        return self.message % dict(log_config=self.log_config, err_msg=self.err_msg)