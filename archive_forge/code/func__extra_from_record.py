from __future__ import absolute_import
import logging
from fnmatch import fnmatch
from sentry_sdk.hub import Hub
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk._compat import iteritems, utc_from_timestamp
from sentry_sdk._types import TYPE_CHECKING
def _extra_from_record(self, record):
    return {k: v for k, v in iteritems(vars(record)) if k not in self.COMMON_RECORD_ATTRS and (not isinstance(k, str) or not k.startswith('_'))}