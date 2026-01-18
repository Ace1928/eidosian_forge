from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
from ..core.has_props import HasProps, Qualified
from ..util.dataclasses import entries, is_dataclass
class HasDocumentRef:
    _document: Document | None
    _temp_document: Document | None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._document = None
        self._temp_document = None

    @property
    def document(self) -> Document | None:
        """ The |Document| this model is attached to (can be ``None``)

        """
        if self._temp_document is not None:
            return self._temp_document
        return self._document

    @document.setter
    def document(self, doc: Document) -> None:
        self._document = doc