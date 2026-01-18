import hashlib
from functools import cached_property
from inspect import isawaitable
from sentry_sdk import configure_scope, start_span
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _sentry_patched_sync_view_handle_errors(self, errors, response_data):
    old_sync_view_handle_errors(self, errors, response_data)
    _sentry_patched_handle_errors(self, errors, response_data)