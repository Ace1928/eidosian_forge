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
def _fetch_pending_client_report(self, force=False, interval=60):
    if not self.options['send_client_reports']:
        return None
    if not (force or self._last_client_report_sent < time.time() - interval):
        return None
    discarded_events = self._discarded_events
    self._discarded_events = defaultdict(int)
    self._last_client_report_sent = time.time()
    if not discarded_events:
        return None
    return Item(PayloadRef(json={'timestamp': time.time(), 'discarded_events': [{'reason': reason, 'category': category, 'quantity': quantity} for (category, reason), quantity in discarded_events.items()]}), type='client_report')