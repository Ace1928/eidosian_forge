from __future__ import division
import sys as _sys
import datetime as _datetime
import uuid as _uuid
import traceback as _traceback
import os as _os
import logging as _logging
from syslog import (LOG_EMERG, LOG_ALERT, LOG_CRIT, LOG_ERR,
from ._journal import __version__, sendv, stream_fd
from ._reader import (_Reader, NOP, APPEND, INVALIDATE,
from . import id128 as _id128
class JournalHandler(_logging.Handler):
    """Journal handler class for the Python logging framework.

    Please see the Python logging module documentation for an overview:
    http://docs.python.org/library/logging.html.

    To create a custom logger whose messages go only to journal:

    >>> import logging
    >>> log = logging.getLogger('custom_logger_name')
    >>> log.propagate = False
    >>> log.addHandler(JournalHandler())
    >>> log.warning("Some message: %s", 'detail')

    Note that by default, message levels `INFO` and `DEBUG` are ignored by the
    logging framework. To enable those log levels:

    >>> log.setLevel(logging.DEBUG)

    To redirect all logging messages to journal regardless of where they come
    from, attach it to the root logger:

    >>> logging.root.addHandler(JournalHandler())

    For more complex configurations when using `dictConfig` or `fileConfig`,
    specify `systemd.journal.JournalHandler` as the handler class.  Only
    standard handler configuration options are supported: `level`, `formatter`,
    `filters`.

    To attach journal MESSAGE_ID, an extra field is supported:

    >>> import uuid
    >>> mid = uuid.UUID('0123456789ABCDEF0123456789ABCDEF')
    >>> log.warning("Message with ID", extra={'MESSAGE_ID': mid})

    Fields to be attached to all messages sent through this handler can be
    specified as keyword arguments. This probably makes sense only for
    SYSLOG_IDENTIFIER and similar fields which are constant for the whole
    program:

    >>> JournalHandler(SYSLOG_IDENTIFIER='my-cool-app')
    <...JournalHandler ...>

    The following journal fields will be sent: `MESSAGE`, `PRIORITY`,
    `THREAD_NAME`, `CODE_FILE`, `CODE_LINE`, `CODE_FUNC`, `LOGGER` (name as
    supplied to getLogger call), `MESSAGE_ID` (optional, see above),
    `SYSLOG_IDENTIFIER` (defaults to sys.argv[0]).

    The function used to actually send messages can be overridden using
    the `sender_function` parameter.
    """

    def __init__(self, level=_logging.NOTSET, sender_function=send, **kwargs):
        super(JournalHandler, self).__init__(level)
        for name in kwargs:
            if not _valid_field_name(name):
                raise ValueError('Invalid field name: ' + name)
        if 'SYSLOG_IDENTIFIER' not in kwargs:
            kwargs['SYSLOG_IDENTIFIER'] = _sys.argv[0]
        self.send = sender_function
        self._extra = kwargs

    @classmethod
    def with_args(cls, config=None):
        """Create a JournalHandler with a configuration dictionary

        This creates a JournalHandler instance, but accepts the parameters through
        a dictionary that can be specified as a positional argument. This is useful
        in contexts like logging.config.fileConfig, where the syntax does not allow
        for positional arguments.

        >>> JournalHandler.with_args({'SYSLOG_IDENTIFIER':'my-cool-app'})
        <...JournalHandler ...>
        """
        return cls(**config or {})

    def emit(self, record):
        """Write `record` as a journal event.

        MESSAGE is taken from the message provided by the user, and PRIORITY,
        LOGGER, THREAD_NAME, CODE_{FILE,LINE,FUNC} fields are appended
        automatically. In addition, record.MESSAGE_ID will be used if present.
        """
        try:
            msg = self.format(record)
            pri = self.map_priority(record.levelno)
            extras = self._extra.copy()
            if record.exc_text:
                extras['EXCEPTION_TEXT'] = record.exc_text
            if record.exc_info:
                extras['EXCEPTION_INFO'] = record.exc_info
            if record.args:
                extras['CODE_ARGS'] = str(record.args)
            extras.update(record.__dict__)
            self.send(msg, PRIORITY=format(pri), LOGGER=record.name, THREAD_NAME=record.threadName, PROCESS_NAME=record.processName, CODE_FILE=record.pathname, CODE_LINE=record.lineno, CODE_FUNC=record.funcName, **extras)
        except Exception:
            self.handleError(record)

    @staticmethod
    def map_priority(levelno):
        """Map logging levels to journald priorities.

        Since Python log level numbers are "sparse", we have to map numbers in
        between the standard levels too.
        """
        if levelno <= _logging.DEBUG:
            return LOG_DEBUG
        elif levelno <= _logging.INFO:
            return LOG_INFO
        elif levelno <= _logging.WARNING:
            return LOG_WARNING
        elif levelno <= _logging.ERROR:
            return LOG_ERR
        elif levelno <= _logging.CRITICAL:
            return LOG_CRIT
        else:
            return LOG_ALERT
    mapPriority = map_priority