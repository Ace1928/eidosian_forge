import warnings
import weakref
from collections import UserDict
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import srsly
from ..errors import Errors, Warnings
from .span_group import SpanGroup
def _ensure_doc(self) -> 'Doc':
    doc = self.doc_ref()
    if doc is None:
        raise ValueError(Errors.E866)
    return doc