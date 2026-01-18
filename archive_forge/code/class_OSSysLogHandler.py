import inspect
import logging
import logging.config
import logging.handlers
import os
class OSSysLogHandler(logging.Handler):
    """Syslog based handler. Only available on UNIX-like platforms."""

    def __init__(self, facility=None):
        facility = facility if facility is not None else syslog.LOG_USER
        if not syslog:
            raise RuntimeError('Syslog not available on this platform')
        logging.Handler.__init__(self)
        binary_name = _get_binary_name()
        syslog.openlog(binary_name, 0, facility)

    def emit(self, record):
        priority = SYSLOG_MAP.get(record.levelname, 7)
        message = self.format(record)
        syslog.syslog(priority, message)