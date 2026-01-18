from __future__ import annotations
import logging # isort:skip
import gc
import weakref
from json import loads
from typing import TYPE_CHECKING, Any, Iterable
from jinja2 import Template
from ..core.enums import HoldPolicyType
from ..core.has_props import is_DataModel
from ..core.query import find, is_single_string_selector
from ..core.serialization import (
from ..core.templates import FILE
from ..core.types import ID
from ..core.validation import check_integrity, process_validation_issues
from ..events import Event
from ..model import Model
from ..themes import Theme, built_in_themes, default as default_theme
from ..util.serialization import make_id
from ..util.strings import nice_join
from ..util.version import __version__
from .callbacks import (
from .events import (
from .json import DocJson, PatchJson
from .models import DocumentModelManager
from .modules import DocumentModuleManager
def _destructively_move(self, dest_doc: Document) -> None:
    """ Move all data in this doc to the dest_doc, leaving this doc empty.

        Args:
            dest_doc (Document) :
                The Bokeh document to populate with data from this one

        Returns:
            None

        """
    if dest_doc is self:
        raise RuntimeError('Attempted to overwrite a document with itself')
    dest_doc.clear()
    roots: list[Model] = []
    with self.models.freeze():
        while self.roots:
            r = next(iter(self.roots))
            self.remove_root(r)
            roots.append(r)
    for r in roots:
        if r.document is not None:
            raise RuntimeError("Somehow we didn't detach %r" % r)
    if len(self.models) != 0:
        raise RuntimeError(f'_all_models still had stuff in it: {self.models!r}')
    for r in roots:
        dest_doc.add_root(r)
    dest_doc.title = self.title