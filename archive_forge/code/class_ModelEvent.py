from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class ModelEvent(Event):
    """ Base class for all Bokeh Model events.

    This base class is not typically useful to instantiate on its own.

    """
    model: Model | None

    def __init__(self, model: Model | None) -> None:
        """ Create a new base event.

        Args:

            model (Model) : a Bokeh model to register event callbacks on

        """
        self.model = model