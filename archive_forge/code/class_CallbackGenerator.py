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
class CallbackGenerator(object):
    error = True

    def __init__(self, root_model: 'Model', link: 'Link', source: 'Reactive', target: Optional['JSLinkTarget']=None, arg_overrides: Dict[str, Any]={}):
        self.root_model = root_model
        self.link = link
        self.source = source
        self.target = target
        self.arg_overrides = arg_overrides
        self.validate()
        specs = self._get_specs(link, source, target)
        for src_spec, tgt_spec, code in specs:
            if src_spec[1] and target is not None and src_spec[1].startswith('event:') and (not tgt_spec[1]):
                continue
            try:
                self._init_callback(root_model, link, source, src_spec, target, tgt_spec, code)
            except Exception:
                if self.error:
                    raise
                else:
                    pass

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

    def _init_callback(self, root_model: 'Model', link: 'Link', source: 'Reactive', src_spec: 'SourceModelSpec', target: 'JSLinkTarget' | None, tgt_spec: 'TargetModelSpec', code: Optional[str]) -> None:
        references = {k: v for k, v in link.param.values().items() if k not in ('source', 'target', 'name', 'code', 'args')}
        src_model = self._resolve_model(root_model, source, src_spec[0])
        if src_model is None:
            return
        ref = root_model.ref['id']
        link_id = (id(link), src_spec, tgt_spec)
        callbacks = list(src_model.js_property_callbacks.values()) + list(src_model.js_event_callbacks.values())
        if any((link_id in cb.tags for cbs in callbacks for cb in cbs)):
            return
        references['source'] = src_model
        tgt_model = None
        if link._requires_target:
            tgt_model = self._resolve_model(root_model, target, tgt_spec[0])
            if tgt_model is not None:
                references['target'] = tgt_model
        for k, v in dict(link.args, **self.arg_overrides).items():
            arg_model = self._resolve_model(root_model, v, None)
            if arg_model is not None:
                references[k] = arg_model
            elif not isinstance(v, param.Parameterized):
                references[k] = v
        if 'holoviews' in sys.modules:
            from .pane.holoviews import HoloViews, is_bokeh_element_plot
            if isinstance(source, HoloViews):
                src = source._plots[ref][0]
            else:
                src = source
            prefix = 'source_' if hasattr(link, 'target') else ''
            if is_bokeh_element_plot(src):
                for k, v in src.handles.items():
                    k = prefix + k
                    if isinstance(v, BkModel) and k not in references:
                        references[k] = v
            if isinstance(target, HoloViews) and ref in target._plots:
                tgt = target._plots[ref][0]
            else:
                tgt = target
            if is_bokeh_element_plot(tgt):
                for k, v in tgt.handles.items():
                    k = 'target_' + k
                    if isinstance(v, BkModel) and k not in references:
                        references[k] = v
        if isinstance(src_model, ReactiveHTML):
            if src_spec[1] in src_model.data.properties():
                references['source'] = src_model = src_model.data
        if isinstance(tgt_model, ReactiveHTML):
            if tgt_spec[1] in tgt_model.data.properties():
                references['target'] = tgt_model = tgt_model.data
        self._initialize_models(link, source, src_model, src_spec[1], target, tgt_model, tgt_spec[1])
        self._process_references(references)
        if code is None:
            code = self._get_code(link, source, src_spec[1], target, tgt_spec[1])
        else:
            code = 'try {{ {code} }} catch(err) {{ console.log(err) }}'.format(code=code)
        src_cb = CustomJS(args=references, code=code, tags=[link_id])
        changes, events = self._get_triggers(link, src_spec)
        for ch in changes:
            src_model.js_on_change(ch, src_cb)
        for ev in events:
            src_model.js_on_event(ev, src_cb)
        tgt_prop = tgt_spec[1]
        if not getattr(link, 'bidirectional', False) or tgt_model is None or tgt_prop is None:
            return
        code = self._get_code(link, target, tgt_prop, source, src_spec[1])
        reverse_references = dict(references)
        reverse_references['source'] = tgt_model
        reverse_references['target'] = src_model
        tgt_cb = CustomJS(args=reverse_references, code=code, tags=[link_id])
        changes, events = self._get_triggers(link, (None, tgt_prop))
        properties = tgt_model.properties()
        for ch in changes:
            if ch not in properties:
                msg = f"Could not link non-existent property '{ch}' on {tgt_model} model"
                if self.error:
                    raise ValueError(msg)
                else:
                    warn(msg)
            tgt_model.js_on_change(ch, tgt_cb)
        for ev in events:
            tgt_model.js_on_event(ev, tgt_cb)

    def _process_references(self, references):
        """
        Method to process references in place.
        """

    def _get_specs(self, link: 'Link', source: 'Reactive', target: 'JSLinkTarget') -> Sequence[Tuple['SourceModelSpec', 'TargetModelSpec', str | None]]:
        """
        Return a list of spec tuples that define source and target
        models.
        """
        return []

    def _get_code(self, link: 'Link', source: 'JSLinkTarget', src_spec: str, target: 'JSLinkTarget' | None, tgt_spec: str | None) -> str:
        """
        Returns the code to be executed.
        """
        return ''

    def _get_triggers(self, link: 'Link', src_spec: 'SourceModelSpec') -> Tuple[List[str], List[str]]:
        """
        Returns the changes and events that trigger the callback.
        """
        return ([], [])

    def _initialize_models(self, link, source: 'Reactive', src_model: 'Model', src_spec: str, target: 'JSLinkTarget' | None, tgt_model: 'Model' | None, tgt_spec: str | None) -> None:
        """
        Applies any necessary initialization to the source and target
        models.
        """

    def validate(self) -> None:
        pass