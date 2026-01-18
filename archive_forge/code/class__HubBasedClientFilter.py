import sys
import logging
from sentry_sdk import utils
from sentry_sdk.hub import Hub
from sentry_sdk.utils import logger
from sentry_sdk.client import _client_init_debug
from logging import LogRecord
class _HubBasedClientFilter(logging.Filter):

    def filter(self, record):
        if _client_init_debug.get(False):
            return True
        hub = Hub.current
        if hub is not None and hub.client is not None:
            return hub.client.options['debug']
        return False