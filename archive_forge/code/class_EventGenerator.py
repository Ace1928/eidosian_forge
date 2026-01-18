import unittest
from collections import namedtuple
from bpython.curtsies import combined_events
from bpython.test import FixLanguageTestCase as TestCase
import curtsies.events
class EventGenerator:

    def __init__(self, initial_events=(), scheduled_events=()):
        self._events = []
        self._current_tick = 0
        for e in initial_events:
            self.schedule_event(e, 0)
        for e, w in scheduled_events:
            self.schedule_event(e, w)

    def schedule_event(self, event, when):
        self._events.append(ScheduledEvent(when, event))
        self._events.sort()

    def send(self, timeout=None):
        if timeout not in [None, 0]:
            raise ValueError('timeout value %r not supported' % timeout)
        if not self._events:
            return None
        if self._events[0].when <= self._current_tick:
            return self._events.pop(0).event
        if timeout == 0:
            return None
        elif timeout is None:
            e = self._events.pop(0)
            self._current_tick = e.when
            return e.event
        else:
            raise ValueError('timeout value %r not supported' % timeout)

    def tick(self, dt=1):
        self._current_tick += dt
        return self._current_tick