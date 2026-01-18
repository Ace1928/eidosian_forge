from __future__ import annotations
import pickle
from io import BytesIO
from typing import (
from rdflib.events import Dispatcher, Event
class StoreCreatedEvent(Event):
    """
    This event is fired when the Store is created, it has the following
    attribute:

      - ``configuration``: string used to create the store

    """