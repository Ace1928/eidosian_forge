from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
def _document_model_changed(self, event: ModelChangedEvent) -> None:
    ...