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
def _send_event(self, event):
    if self._check_disabled('error'):
        self.on_dropped_event('self_rate_limits')
        self.record_lost_event('ratelimit_backoff', data_category='error')
        return None
    body = io.BytesIO()
    if self._compresslevel == 0:
        body.write(json_dumps(event))
    else:
        with gzip.GzipFile(fileobj=body, mode='w', compresslevel=self._compresslevel) as f:
            f.write(json_dumps(event))
    assert self.parsed_dsn is not None
    logger.debug('Sending event, type:%s level:%s event_id:%s project:%s host:%s' % (event.get('type') or 'null', event.get('level') or 'null', event.get('event_id') or 'null', self.parsed_dsn.project_id, self.parsed_dsn.host))
    headers = {'Content-Type': 'application/json'}
    if self._compresslevel > 0:
        headers['Content-Encoding'] = 'gzip'
    self._send_request(body.getvalue(), headers=headers)
    return None