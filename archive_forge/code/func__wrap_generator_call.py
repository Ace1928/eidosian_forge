from __future__ import absolute_import
import sys
import types
from sentry_sdk._functools import wraps
from sentry_sdk.hub import Hub
from sentry_sdk._compat import reraise
from sentry_sdk.utils import capture_internal_exceptions, event_from_exception
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._types import TYPE_CHECKING
def _wrap_generator_call(gen, client):
    """
    Wrap the generator to handle any failures.
    """
    while True:
        try:
            yield next(gen)
        except StopIteration:
            break
        except Exception:
            raise_exception(client)