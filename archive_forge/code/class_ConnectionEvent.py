from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class ConnectionEvent(DocumentEvent):
    """ Base class for connection status related events.

    """