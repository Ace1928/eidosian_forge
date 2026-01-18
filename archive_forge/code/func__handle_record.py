from __future__ import absolute_import
import logging
from fnmatch import fnmatch
from sentry_sdk.hub import Hub
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk._compat import iteritems, utc_from_timestamp
from sentry_sdk._types import TYPE_CHECKING
def _handle_record(self, record):
    if self._handler is not None and record.levelno >= self._handler.level:
        self._handler.handle(record)
    if self._breadcrumb_handler is not None and record.levelno >= self._breadcrumb_handler.level:
        self._breadcrumb_handler.handle(record)