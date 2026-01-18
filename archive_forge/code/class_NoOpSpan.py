import uuid
import random
from datetime import datetime, timedelta
import sentry_sdk
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.utils import is_valid_sample_rate, logger, nanosecond_time
from sentry_sdk._compat import datetime_utcnow, utc_from_timestamp, PY2
from sentry_sdk.consts import SPANDATA
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing_utils import (
from sentry_sdk.metrics import LocalAggregator
class NoOpSpan(Span):

    def __repr__(self):
        return self.__class__.__name__

    @property
    def containing_transaction(self):
        return None

    def start_child(self, instrumenter=INSTRUMENTER.SENTRY, **kwargs):
        return NoOpSpan()

    def new_span(self, **kwargs):
        return self.start_child(**kwargs)

    def to_traceparent(self):
        return ''

    def to_baggage(self):
        return None

    def get_baggage(self):
        return None

    def iter_headers(self):
        return iter(())

    def set_tag(self, key, value):
        pass

    def set_data(self, key, value):
        pass

    def set_status(self, value):
        pass

    def set_http_status(self, http_status):
        pass

    def is_success(self):
        return True

    def to_json(self):
        return {}

    def get_trace_context(self):
        return {}

    def finish(self, hub=None, end_timestamp=None):
        pass

    def set_measurement(self, name, value, unit=''):
        pass

    def set_context(self, key, value):
        pass

    def init_span_recorder(self, maxlen):
        pass

    def _set_initial_sampling_decision(self, sampling_context):
        pass