from __future__ import annotations
import logging # isort:skip
from typing import (
from ..core.serialization import Serializable, Serializer
from .json import (
class DocumentChangedMixin:

    def _document_changed(self, event: DocumentChangedEvent) -> None:
        ...