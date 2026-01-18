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
def _send_envelope(self, envelope):
    new_items = []
    for item in envelope.items:
        if self._check_disabled(item.data_category):
            if item.data_category in ('transaction', 'error', 'default'):
                self.on_dropped_event('self_rate_limits')
            self.record_lost_event('ratelimit_backoff', item=item)
        else:
            new_items.append(item)
    envelope = Envelope(headers=envelope.headers, items=new_items)
    if not envelope.items:
        return None
    client_report_item = self._fetch_pending_client_report(interval=30)
    if client_report_item is not None:
        envelope.items.append(client_report_item)
    body = io.BytesIO()
    if self._compresslevel == 0:
        envelope.serialize_into(body)
    else:
        with gzip.GzipFile(fileobj=body, mode='w', compresslevel=self._compresslevel) as f:
            envelope.serialize_into(f)
    assert self.parsed_dsn is not None
    logger.debug('Sending envelope [%s] project:%s host:%s', envelope.description, self.parsed_dsn.project_id, self.parsed_dsn.host)
    headers = {'Content-Type': 'application/x-sentry-envelope'}
    if self._compresslevel > 0:
        headers['Content-Encoding'] = 'gzip'
    self._send_request(body.getvalue(), headers=headers, endpoint_type='envelope', envelope=envelope)
    return None