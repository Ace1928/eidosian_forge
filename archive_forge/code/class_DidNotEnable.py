from __future__ import absolute_import
from threading import Lock
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import logger
class DidNotEnable(Exception):
    """
    The integration could not be enabled due to a trivial user error like
    `flask` not being installed for the `FlaskIntegration`.

    This exception is silently swallowed for default integrations, but reraised
    for explicitly enabled integrations.
    """