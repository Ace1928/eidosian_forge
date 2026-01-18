import copy
import sys
from contextlib import contextmanager
from sentry_sdk._compat import with_metaclass
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.scope import Scope
from sentry_sdk.client import Client
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._types import TYPE_CHECKING
def _check_python_deprecations():
    version = sys.version_info[:2]
    if version == (3, 4) or version == (3, 5):
        logger.warning('sentry-sdk 2.0.0 will drop support for Python %s.', '{}.{}'.format(*version))
        logger.warning('Please upgrade to the latest version to continue receiving upgrades and bugfixes.')