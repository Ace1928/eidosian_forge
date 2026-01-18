import configparser
import logging
import logging.handlers
import os
import signal
import sys
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
class RootwrapConfig(object):

    def __init__(self, config):
        self.filters_path = config.get('DEFAULT', 'filters_path').split(',')
        if config.has_option('DEFAULT', 'exec_dirs'):
            self.exec_dirs = config.get('DEFAULT', 'exec_dirs').split(',')
        else:
            self.exec_dirs = []
            if 'PATH' in os.environ:
                self.exec_dirs = os.environ['PATH'].split(':')
        if config.has_option('DEFAULT', 'syslog_log_facility'):
            v = config.get('DEFAULT', 'syslog_log_facility')
            facility_names = logging.handlers.SysLogHandler.facility_names
            self.syslog_log_facility = getattr(logging.handlers.SysLogHandler, v, None)
            if self.syslog_log_facility is None and v in facility_names:
                self.syslog_log_facility = facility_names.get(v)
            if self.syslog_log_facility is None:
                raise ValueError('Unexpected syslog_log_facility: %s' % v)
        else:
            default_facility = logging.handlers.SysLogHandler.LOG_SYSLOG
            self.syslog_log_facility = default_facility
        if config.has_option('DEFAULT', 'syslog_log_level'):
            v = config.get('DEFAULT', 'syslog_log_level')
            level = v.upper()
            if hasattr(logging, '_nameToLevel') and level in logging._nameToLevel:
                self.syslog_log_level = logging._nameToLevel[level]
            else:
                self.syslog_log_level = logging.getLevelName(level)
            if self.syslog_log_level == 'Level %s' % level:
                raise ValueError('Unexpected syslog_log_level: %r' % v)
        else:
            self.syslog_log_level = logging.ERROR
        if config.has_option('DEFAULT', 'use_syslog'):
            self.use_syslog = config.getboolean('DEFAULT', 'use_syslog')
        else:
            self.use_syslog = False
        if config.has_option('DEFAULT', 'daemon_timeout'):
            self.daemon_timeout = int(config.get('DEFAULT', 'daemon_timeout'))
        else:
            self.daemon_timeout = 600
        if config.has_option('DEFAULT', 'rlimit_nofile'):
            self.rlimit_nofile = int(config.get('DEFAULT', 'rlimit_nofile'))
        else:
            self.rlimit_nofile = 1024