from __future__ import absolute_import
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
def _add_user_to_event(event):
    if flask_login is None:
        return
    user = flask_login.current_user
    if user is None:
        return
    with capture_internal_exceptions():
        user_info = event.setdefault('user', {})
        try:
            user_info.setdefault('id', user.get_id())
        except AttributeError:
            pass
        try:
            user_info.setdefault('email', user.email)
        except Exception:
            pass
        try:
            user_info.setdefault('username', user.username)
        except Exception:
            pass