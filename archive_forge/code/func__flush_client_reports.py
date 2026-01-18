from __future__ import print_function
import io
import gzip
import time
from datetime import timedelta
from collections import defaultdict
import urllib3
import certifi
from sentry_sdk.utils import Dsn, logger, capture_internal_exceptions, json_dumps
from sentry_sdk.worker import BackgroundWorker
from sentry_sdk.envelope import Envelope, Item, PayloadRef
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk._types import TYPE_CHECKING
def _flush_client_reports(self, force=False):
    client_report = self._fetch_pending_client_report(force=force, interval=60)
    if client_report is not None:
        self.capture_envelope(Envelope(items=[client_report]))