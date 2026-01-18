from __future__ import annotations
import pickle
from io import BytesIO
from typing import (
from rdflib.events import Dispatcher, Event
class TripleRemovedEvent(Event):
    """
    This event is fired when a triple is removed, it has the following
    attributes:

      - the ``triple`` removed from the graph
      - the ``context`` of the triple, if any
      - the ``graph`` from which the triple was removed
    """