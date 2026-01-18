import os
import time
from threading import Thread, Lock
from contextlib import contextmanager
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.session import Session
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import format_timestamp
def add_aggregate_session(self, session):
    with self._aggregate_lock:
        attrs = session.get_json_attrs(with_user_info=False)
        primary_key = tuple(sorted(attrs.items()))
        secondary_key = session.truncated_started
        states = self.pending_aggregates.setdefault(primary_key, {})
        state = states.setdefault(secondary_key, {})
        if 'started' not in state:
            state['started'] = format_timestamp(session.truncated_started)
        if session.status == 'crashed':
            state['crashed'] = state.get('crashed', 0) + 1
        elif session.status == 'abnormal':
            state['abnormal'] = state.get('abnormal', 0) + 1
        elif session.errors > 0:
            state['errored'] = state.get('errored', 0) + 1
        else:
            state['exited'] = state.get('exited', 0) + 1