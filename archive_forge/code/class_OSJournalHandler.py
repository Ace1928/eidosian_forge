import inspect
import logging
import logging.config
import logging.handlers
import os
class OSJournalHandler(logging.Handler):
    custom_fields = ('project_name', 'project_id', 'user_name', 'user_id', 'request_id')

    def __init__(self, facility=None):
        if not journal:
            raise RuntimeError('Systemd bindings do not exist')
        if not facility:
            if not syslog:
                raise RuntimeError('syslog is not available on this platform')
            facility = syslog.LOG_USER
        logging.Handler.__init__(self)
        self.binary_name = _get_binary_name()
        self.facility = facility

    def emit(self, record):
        priority = SYSLOG_MAP.get(record.levelname, 7)
        message = self.format(record)
        extras = {'CODE_FILE': record.pathname, 'CODE_LINE': record.lineno, 'CODE_FUNC': record.funcName, 'THREAD_NAME': record.threadName, 'PROCESS_NAME': record.processName, 'LOGGER_NAME': record.name, 'LOGGER_LEVEL': record.levelname, 'SYSLOG_IDENTIFIER': self.binary_name, 'PRIORITY': priority, 'SYSLOG_FACILITY': self.facility}
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatter.formatException(record.exc_info)
        if record.exc_text:
            extras['EXCEPTION_INFO'] = record.exc_text
            extras['EXCEPTION_TEXT'] = record.exc_text
        for field in self.custom_fields:
            value = record.__dict__.get(field)
            if value:
                extras[field.upper()] = value
        journal.send(message, **extras)