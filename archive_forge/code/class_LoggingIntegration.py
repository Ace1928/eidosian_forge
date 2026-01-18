from __future__ import absolute_import
import logging
from fnmatch import fnmatch
from sentry_sdk.hub import Hub
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk._compat import iteritems, utc_from_timestamp
from sentry_sdk._types import TYPE_CHECKING
class LoggingIntegration(Integration):
    identifier = 'logging'

    def __init__(self, level=DEFAULT_LEVEL, event_level=DEFAULT_EVENT_LEVEL):
        self._handler = None
        self._breadcrumb_handler = None
        if level is not None:
            self._breadcrumb_handler = BreadcrumbHandler(level=level)
        if event_level is not None:
            self._handler = EventHandler(level=event_level)

    def _handle_record(self, record):
        if self._handler is not None and record.levelno >= self._handler.level:
            self._handler.handle(record)
        if self._breadcrumb_handler is not None and record.levelno >= self._breadcrumb_handler.level:
            self._breadcrumb_handler.handle(record)

    @staticmethod
    def setup_once():
        old_callhandlers = logging.Logger.callHandlers

        def sentry_patched_callhandlers(self, record):
            ignored_loggers = _IGNORED_LOGGERS
            try:
                return old_callhandlers(self, record)
            finally:
                if ignored_loggers is not None and record.name not in ignored_loggers:
                    integration = Hub.current.get_integration(LoggingIntegration)
                    if integration is not None:
                        integration._handle_record(record)
        logging.Logger.callHandlers = sentry_patched_callhandlers