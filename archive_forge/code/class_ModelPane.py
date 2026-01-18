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
class ModelPane(PaneBase):
    """
    ModelPane provides a baseclass that allows quickly wrapping a
    Bokeh model and translating parameters defined on the class
    with properties defined on the model.

    In simple cases subclasses only have to define the Bokeh model to
    render to and the `_transform_object` method which transforms the
    Python object being wrapped into properties that the
    `bokeh.model.Model` can consume.
    """
    _bokeh_model: ClassVar[Model]
    __abstract = True

    def _get_model(self, doc: Document, root: Model | None=None, parent: Model | None=None, comm: Comm | None=None) -> Model:
        model = self._bokeh_model(**self._get_properties(doc))
        if root is None:
            root = model
        self._models[root.ref['id']] = (model, parent)
        self._link_props(model, self._linked_properties, doc, root, comm)
        return model

    def _update(self, ref: str, model: Model) -> None:
        model.update(**self._get_properties(model.document))

    def _init_params(self):
        params = {p: v for p, v in self.param.values().items() if v is not None and p not in ('name', 'default_layout')}
        params['object'] = self.object
        return params

    def _transform_object(self, obj: Any) -> Dict[str, Any]:
        return dict(object=obj)

    def _process_param_change(self, params):
        if 'object' in params:
            params.update(self._transform_object(params.pop('object')))
        if self._bokeh_model is not None and 'stylesheets' in params:
            css = getattr(self._bokeh_model, '__css__', [])
            params['stylesheets'] = [ImportedStyleSheet(url=ss) for ss in css] + params['stylesheets']
        return super()._process_param_change(params)