from __future__ import annotations
import difflib
import sys
import weakref
from typing import (
import param
from bokeh.models import CustomJS, LayoutDOM, Model as BkModel
from .io.datamodel import create_linked_datamodel
from .io.loading import LOADING_INDICATOR_CSS_CLASS
from .models import ReactiveHTML
from .reactive import Reactive
from .util.warnings import warn
from .viewable import Viewable
@classmethod
def _resolve_model(cls, root_model: 'Model', obj: 'JSLinkTarget', model_spec: str | None) -> 'Model' | None:
    """
        Resolves a model given the supplied object and a model_spec.

        Arguments
        ----------
        root_model: bokeh.model.Model
          The root bokeh model often used to index models
        obj: holoviews.plotting.ElementPlot or bokeh.model.Model or panel.Viewable
          The object to look the model up on
        model_spec: string
          A string defining how to look up the model, can be a single
          string defining the handle in a HoloViews plot or a path
          split by periods (.) to indicate a multi-level lookup.

        Returns
        -------
        model: bokeh.model.Model
          The resolved bokeh model
        """
    from .pane.holoviews import is_bokeh_element_plot
    model = None
    if 'holoviews' in sys.modules and is_bokeh_element_plot(obj):
        if model_spec is None:
            return obj.state
        else:
            model_specs = model_spec.split('.')
            handle_spec = model_specs[0]
            if len(model_specs) > 1:
                model_spec = '.'.join(model_specs[1:])
            else:
                model_spec = None
            model = obj.handles[handle_spec]
    elif isinstance(obj, Viewable):
        model, _ = obj._models.get(root_model.ref['id'], (None, None))
    elif isinstance(obj, BkModel):
        model = obj
    elif isinstance(obj, param.Parameterized):
        model = create_linked_datamodel(obj, root_model)
    if model_spec is not None:
        for spec in model_spec.split('.'):
            model = getattr(model, spec)
    return model