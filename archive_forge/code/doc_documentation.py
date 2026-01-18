from __future__ import annotations
import logging # isort:skip
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, cast
from ..document import Document
from .state import curstate
 Configure the current document (returned by curdoc()).

    Args:
        doc (Document) : new Document to use for curdoc()

    Returns:
        None

    .. warning::
        Calling this function will replace any existing document.

    