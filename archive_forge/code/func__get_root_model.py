from __future__ import annotations
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import ImportedStyleSheet
from bokeh.models.layouts import (
from .._param import Margin
from ..io.cache import _generate_hash
from ..io.document import create_doc_if_none_exists, unlocked
from ..io.notebook import push
from ..io.state import state
from ..layout.base import (
from ..links import Link
from ..models import ReactiveHTML as _BkReactiveHTML
from ..reactive import Reactive
from ..util import param_reprs, param_watchers
from ..util.checks import is_dataframe, is_series
from ..util.parameters import get_params_to_inherit
from ..viewable import (
def _get_root_model(self, doc: Optional[Document]=None, comm: Comm | None=None, preprocess: bool=True) -> Tuple[Viewable, Model]:
    if self._updates:
        root = self._get_model(doc, comm=comm)
        root_view = self
    else:
        root = self.layout._get_model(doc, comm=comm)
        root_view = self.layout
    if preprocess:
        self._preprocess(root)
    return (root_view, root)