from __future__ import annotations
import pickle
from io import BytesIO
from typing import (
from rdflib.events import Dispatcher, Event
class TripleAddedEvent(Event):
    """
    This event is fired when a triple is added, it has the following
    attributes:

      - the ``triple`` added to the graph
      - the ``context`` of the triple, if any
      - the ``graph`` to which the triple was added
    """