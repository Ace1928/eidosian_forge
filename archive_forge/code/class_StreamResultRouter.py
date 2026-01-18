import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
class StreamResultRouter(StreamResult):
    """A StreamResult that routes events.

    StreamResultRouter forwards received events to another StreamResult object,
    selected by a dynamic forwarding policy. Events where no destination is
    found are forwarded to the fallback StreamResult, or an error is raised.

    Typical use is to construct a router with a fallback and then either
    create up front mapping rules, or create them as-needed from the fallback
    handler::

      >>> router = StreamResultRouter()
      >>> sink = doubles.StreamResult()
      >>> router.add_rule(sink, 'route_code_prefix', route_prefix='0',
      ...     consume_route=True)
      >>> router.status(
      ...     test_id='foo', route_code='0/1', test_status='uxsuccess')

    StreamResultRouter has no buffering.

    When adding routes (and for the fallback) whether to call startTestRun and
    stopTestRun or to not call them is controllable by passing
    'do_start_stop_run'. The default is to call them for the fallback only.
    If a route is added after startTestRun has been called, and
    do_start_stop_run is True then startTestRun is called immediately on the
    new route sink.

    There is no a-priori defined lookup order for routes: if they are ambiguous
    the behaviour is undefined. Only a single route is chosen for any event.
    """
    _policies = {}

    def __init__(self, fallback=None, do_start_stop_run=True):
        """Construct a StreamResultRouter with optional fallback.

        :param fallback: A StreamResult to forward events to when no route
            exists for them.
        :param do_start_stop_run: If False do not pass startTestRun and
            stopTestRun onto the fallback.
        """
        self.fallback = fallback
        self._route_code_prefixes = {}
        self._test_ids = {}
        self._sinks = []
        if do_start_stop_run and fallback:
            self._sinks.append(fallback)
        self._in_run = False

    def startTestRun(self):
        super().startTestRun()
        for sink in self._sinks:
            sink.startTestRun()
        self._in_run = True

    def stopTestRun(self):
        super().stopTestRun()
        for sink in self._sinks:
            sink.stopTestRun()
        self._in_run = False

    def status(self, **kwargs):
        route_code = kwargs.get('route_code', None)
        test_id = kwargs.get('test_id', None)
        if route_code is not None:
            prefix = route_code.split('/')[0]
        else:
            prefix = route_code
        if prefix in self._route_code_prefixes:
            target, consume_route = self._route_code_prefixes[prefix]
            if route_code is not None and consume_route:
                route_code = route_code[len(prefix) + 1:]
                if not route_code:
                    route_code = None
                kwargs['route_code'] = route_code
        elif test_id in self._test_ids:
            target = self._test_ids[test_id]
        else:
            target = self.fallback
        target.status(**kwargs)

    def add_rule(self, sink, policy, do_start_stop_run=False, **policy_args):
        """Add a rule to route events to sink when they match a given policy.

        :param sink: A StreamResult to receive events.
        :param policy: A routing policy. Valid policies are
            'route_code_prefix' and 'test_id'.
        :param do_start_stop_run: If True then startTestRun and stopTestRun
            events will be passed onto this sink.

        :raises: ValueError if the policy is unknown
        :raises: TypeError if the policy is given arguments it cannot handle.

        ``route_code_prefix`` routes events based on a prefix of the route
        code in the event. It takes a ``route_prefix`` argument to match on
        (e.g. '0') and a ``consume_route`` argument, which, if True, removes
        the prefix from the ``route_code`` when forwarding events.

        ``test_id`` routes events based on the test id.  It takes a single
        argument, ``test_id``.  Use ``None`` to select non-test events.
        """
        policy_method = StreamResultRouter._policies.get(policy, None)
        if not policy_method:
            raise ValueError(f'bad policy {policy!r}')
        policy_method(self, sink, **policy_args)
        if do_start_stop_run:
            self._sinks.append(sink)
        if self._in_run:
            sink.startTestRun()

    def _map_route_code_prefix(self, sink, route_prefix, consume_route=False):
        if '/' in route_prefix:
            raise TypeError(f'{route_prefix!r} is more than one route step long')
        self._route_code_prefixes[route_prefix] = (sink, consume_route)
    _policies['route_code_prefix'] = _map_route_code_prefix

    def _map_test_id(self, sink, test_id):
        self._test_ids[test_id] = sink
    _policies['test_id'] = _map_test_id