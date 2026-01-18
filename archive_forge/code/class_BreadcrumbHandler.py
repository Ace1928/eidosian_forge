from __future__ import absolute_import
import logging
from fnmatch import fnmatch
from sentry_sdk.hub import Hub
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk._compat import iteritems, utc_from_timestamp
from sentry_sdk._types import TYPE_CHECKING
class BreadcrumbHandler(_BaseHandler):
    """
    A logging handler that records breadcrumbs for each log record.

    Note that you do not have to use this class if the logging integration is enabled, which it is by default.
    """

    def emit(self, record):
        with capture_internal_exceptions():
            self.format(record)
            return self._emit(record)

    def _emit(self, record):
        if not self._can_record(record):
            return
        Hub.current.add_breadcrumb(self._breadcrumb_from_record(record), hint={'log_record': record})

    def _breadcrumb_from_record(self, record):
        return {'type': 'log', 'level': self._logging_to_event_level(record), 'category': record.name, 'message': record.message, 'timestamp': utc_from_timestamp(record.created), 'data': self._extra_from_record(record)}