import abc
import argparse
import logging
import uuid
from oslo_config import cfg
from oslo_utils import timeutils
from stevedore import extension
from stevedore import named
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import transport as msg_transport
class _SubNotifier(Notifier):
    _marker = Notifier._marker

    def __init__(self, base, publisher_id, retry):
        self._base = base
        self.transport = base.transport
        self.publisher_id = publisher_id
        self.retry = retry
        self._serializer = self._base._serializer
        self._driver_mgr = self._base._driver_mgr

    def _notify(self, ctxt, event_type, payload, priority):
        super(_SubNotifier, self)._notify(ctxt, event_type, payload, priority)

    @classmethod
    def _prepare(cls, base, publisher_id=_marker, retry=_marker):
        if publisher_id is cls._marker:
            publisher_id = base.publisher_id
        if retry is cls._marker:
            retry = base.retry
        return cls(base, publisher_id, retry=retry)