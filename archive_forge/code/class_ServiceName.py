from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker
class ServiceName(enum.Enum):
    """
    The Background Service that will be associated with the commands/events.
    Every Background Service operates independently, but they share the same
    API.
    """
    BACKGROUND_FETCH = 'backgroundFetch'
    BACKGROUND_SYNC = 'backgroundSync'
    PUSH_MESSAGING = 'pushMessaging'
    NOTIFICATIONS = 'notifications'
    PAYMENT_HANDLER = 'paymentHandler'
    PERIODIC_BACKGROUND_SYNC = 'periodicBackgroundSync'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)