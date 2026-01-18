from __future__ import absolute_import
import enum
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import (
class LoguruIntegration(Integration):
    identifier = 'loguru'

    def __init__(self, level=DEFAULT_LEVEL, event_level=DEFAULT_EVENT_LEVEL, breadcrumb_format=DEFAULT_FORMAT, event_format=DEFAULT_FORMAT):
        global _ADDED_HANDLERS
        breadcrumb_handler, event_handler = _ADDED_HANDLERS
        if breadcrumb_handler is not None:
            logger.remove(breadcrumb_handler)
            breadcrumb_handler = None
        if event_handler is not None:
            logger.remove(event_handler)
            event_handler = None
        if level is not None:
            breadcrumb_handler = logger.add(LoguruBreadcrumbHandler(level=level), level=level, format=breadcrumb_format)
        if event_level is not None:
            event_handler = logger.add(LoguruEventHandler(level=event_level), level=event_level, format=event_format)
        _ADDED_HANDLERS = (breadcrumb_handler, event_handler)

    @staticmethod
    def setup_once():
        pass