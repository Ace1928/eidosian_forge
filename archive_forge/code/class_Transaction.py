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
class Transaction(Span):
    """The Transaction is the root element that holds all the spans
    for Sentry performance instrumentation."""
    __slots__ = ('name', 'source', 'parent_sampled', 'sample_rate', '_measurements', '_contexts', '_profile', '_baggage')

    def __init__(self, name='', parent_sampled=None, baggage=None, source=TRANSACTION_SOURCE_CUSTOM, **kwargs):
        """Constructs a new Transaction.

        :param name: Identifier of the transaction.
            Will show up in the Sentry UI.
        :param parent_sampled: Whether the parent transaction was sampled.
            If True this transaction will be kept, if False it will be discarded.
        :param baggage: The W3C baggage header value.
            (see https://www.w3.org/TR/baggage/)
        :param source: A string describing the source of the transaction name.
            This will be used to determine the transaction's type.
            See https://develop.sentry.dev/sdk/event-payloads/transaction/#transaction-annotations
            for more information. Default "custom".
        """
        if not name and 'transaction' in kwargs:
            logger.warning('Deprecated: use Transaction(name=...) to create transactions instead of Span(transaction=...).')
            name = kwargs.pop('transaction')
        super(Transaction, self).__init__(**kwargs)
        self.name = name
        self.source = source
        self.sample_rate = None
        self.parent_sampled = parent_sampled
        self._measurements = {}
        self._contexts = {}
        self._profile = None
        self._baggage = baggage

    def __repr__(self):
        return '<%s(name=%r, op=%r, trace_id=%r, span_id=%r, parent_span_id=%r, sampled=%r, source=%r)>' % (self.__class__.__name__, self.name, self.op, self.trace_id, self.span_id, self.parent_span_id, self.sampled, self.source)

    def __enter__(self):
        super(Transaction, self).__enter__()
        if self._profile is not None:
            self._profile.__enter__()
        return self

    def __exit__(self, ty, value, tb):
        if self._profile is not None:
            self._profile.__exit__(ty, value, tb)
        super(Transaction, self).__exit__(ty, value, tb)

    @property
    def containing_transaction(self):
        """The root element of the span tree.
        In the case of a transaction it is the transaction itself.
        """
        return self

    def finish(self, hub=None, end_timestamp=None):
        """Finishes the transaction and sends it to Sentry.
        All finished spans in the transaction will also be sent to Sentry.

        :param hub: The hub to use for this transaction.
            If not provided, the current hub will be used.
        :param end_timestamp: Optional timestamp that should
            be used as timestamp instead of the current time.

        :return: The event ID if the transaction was sent to Sentry,
            otherwise None.
        """
        if self.timestamp is not None:
            return None
        hub = hub or self.hub or sentry_sdk.Hub.current
        client = hub.client
        if client is None:
            return None
        if self._span_recorder is None:
            logger.debug('Discarding transaction because sampled = False')
            if client.transport and has_tracing_enabled(client.options):
                if client.monitor and client.monitor.downsample_factor > 0:
                    reason = 'backpressure'
                else:
                    reason = 'sample_rate'
                client.transport.record_lost_event(reason, data_category='transaction')
            return None
        if not self.name:
            logger.warning('Transaction has no name, falling back to `<unlabeled transaction>`.')
            self.name = '<unlabeled transaction>'
        super(Transaction, self).finish(hub, end_timestamp)
        if not self.sampled:
            if self.sampled is None:
                logger.warning('Discarding transaction without sampling decision.')
            return None
        finished_spans = [span.to_json() for span in self._span_recorder.spans if span.timestamp is not None]
        self._span_recorder = None
        contexts = {}
        contexts.update(self._contexts)
        contexts.update({'trace': self.get_trace_context()})
        event = {'type': 'transaction', 'transaction': self.name, 'transaction_info': {'source': self.source}, 'contexts': contexts, 'tags': self._tags, 'timestamp': self.timestamp, 'start_timestamp': self.start_timestamp, 'spans': finished_spans}
        if self._profile is not None and self._profile.valid():
            event['profile'] = self._profile
            self._profile = None
        event['measurements'] = self._measurements
        if self._local_aggregator is not None:
            metrics_summary = self._local_aggregator.to_json()
            if metrics_summary:
                event['_metrics_summary'] = metrics_summary
        return hub.capture_event(event)

    def set_measurement(self, name, value, unit=''):
        self._measurements[name] = {'value': value, 'unit': unit}

    def set_context(self, key, value):
        """Sets a context. Transactions can have multiple contexts
        and they should follow the format described in the "Contexts Interface"
        documentation.

        :param key: The name of the context.
        :param value: The information about the context.
        """
        self._contexts[key] = value

    def set_http_status(self, http_status):
        """Sets the status of the Transaction according to the given HTTP status.

        :param http_status: The HTTP status code."""
        super(Transaction, self).set_http_status(http_status)
        self.set_context('response', {'status_code': http_status})

    def to_json(self):
        """Returns a JSON-compatible representation of the transaction."""
        rv = super(Transaction, self).to_json()
        rv['name'] = self.name
        rv['source'] = self.source
        rv['sampled'] = self.sampled
        return rv

    def get_baggage(self):
        """Returns the :py:class:`~sentry_sdk.tracing_utils.Baggage`
        associated with the Transaction.

        The first time a new baggage with Sentry items is made,
        it will be frozen."""
        if not self._baggage or self._baggage.mutable:
            self._baggage = Baggage.populate_from_transaction(self)
        return self._baggage

    def _set_initial_sampling_decision(self, sampling_context):
        """
        Sets the transaction's sampling decision, according to the following
        precedence rules:

        1. If a sampling decision is passed to `start_transaction`
        (`start_transaction(name: "my transaction", sampled: True)`), that
        decision will be used, regardless of anything else

        2. If `traces_sampler` is defined, its decision will be used. It can
        choose to keep or ignore any parent sampling decision, or use the
        sampling context data to make its own decision or to choose a sample
        rate for the transaction.

        3. If `traces_sampler` is not defined, but there's a parent sampling
        decision, the parent sampling decision will be used.

        4. If `traces_sampler` is not defined and there's no parent sampling
        decision, `traces_sample_rate` will be used.
        """
        hub = self.hub or sentry_sdk.Hub.current
        client = hub.client
        options = client and client.options or {}
        transaction_description = '{op}transaction <{name}>'.format(op='<' + self.op + '> ' if self.op else '', name=self.name)
        if not client or not has_tracing_enabled(options):
            self.sampled = False
            return
        if self.sampled is not None:
            self.sample_rate = float(self.sampled)
            return
        sample_rate = options['traces_sampler'](sampling_context) if callable(options.get('traces_sampler')) else sampling_context['parent_sampled'] if sampling_context['parent_sampled'] is not None else options['traces_sample_rate']
        if not is_valid_sample_rate(sample_rate, source='Tracing'):
            logger.warning('[Tracing] Discarding {transaction_description} because of invalid sample rate.'.format(transaction_description=transaction_description))
            self.sampled = False
            return
        self.sample_rate = float(sample_rate)
        if client.monitor:
            self.sample_rate /= 2 ** client.monitor.downsample_factor
        if not self.sample_rate:
            logger.debug('[Tracing] Discarding {transaction_description} because {reason}'.format(transaction_description=transaction_description, reason='traces_sampler returned 0 or False' if callable(options.get('traces_sampler')) else 'traces_sample_rate is set to 0'))
            self.sampled = False
            return
        self.sampled = random.random() < self.sample_rate
        if self.sampled:
            logger.debug('[Tracing] Starting {transaction_description}'.format(transaction_description=transaction_description))
        else:
            logger.debug("[Tracing] Discarding {transaction_description} because it's not included in the random sample (sampling rate = {sample_rate})".format(transaction_description=transaction_description, sample_rate=self.sample_rate))