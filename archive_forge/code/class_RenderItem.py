from __future__ import annotations
import logging # isort:skip
import re
from contextlib import contextmanager
from typing import (
from weakref import WeakKeyDictionary
from ..core.types import ID
from ..document.document import Document
from ..model import Model, collect_models
from ..settings import settings
from ..themes.theme import Theme
from ..util.dataclasses import dataclass, field
from ..util.serialization import (
class RenderItem:

    def __init__(self, docid: ID | None=None, token: str | None=None, elementid: ID | None=None, roots: list[Model] | dict[Model, ID] | None=None, use_for_title: bool | None=None):
        if docid is None and token is None or (docid is not None and token is not None):
            raise ValueError('either docid or sessionid must be provided')
        if roots is None:
            roots = dict()
        elif isinstance(roots, list):
            roots = {root: make_globally_unique_id() for root in roots}
        self.docid = docid
        self.token = token
        self.elementid = elementid
        self.roots = RenderRoots(roots)
        self.use_for_title = use_for_title

    def to_json(self) -> dict[str, Any]:
        json: dict[str, Any] = {}
        if self.docid is not None:
            json['docid'] = self.docid
        else:
            json['token'] = self.token
        if self.elementid is not None:
            json['elementid'] = self.elementid
        if self.roots:
            json['roots'] = self.roots.to_json()
            json['root_ids'] = [root.id for root in self.roots]
        if self.use_for_title is not None:
            json['use_for_title'] = self.use_for_title
        return json

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.to_json() == other.to_json()