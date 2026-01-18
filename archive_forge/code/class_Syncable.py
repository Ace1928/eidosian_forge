from __future__ import annotations
import datetime as dt
import difflib
import inspect
import logging
import re
import sys
import textwrap
from collections import Counter, defaultdict, namedtuple
from functools import lru_cache, partial
from pprint import pformat
from typing import (
import numpy as np
import param
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.model import DataModel
from bokeh.models import ImportedStyleSheet
from packaging.version import Version
from param.parameterized import (
from .io.document import unlocked
from .io.model import hold
from .io.notebook import push
from .io.resources import (
from .io.state import set_curdoc, state
from .models.reactive_html import (
from .util import (
from .viewable import Layoutable, Renderable, Viewable
class Syncable(Renderable):
    """
    Syncable is an extension of the Renderable object which can not
    only render to a bokeh model but also sync the parameters on the
    object with the properties on the model.

    In order to bi-directionally link parameters with bokeh model
    instances the _link_params and _link_props methods define
    callbacks triggered when either the parameter or bokeh property
    values change. Since there may not be a 1-to-1 mapping between
    parameter and the model property the _process_property_change and
    _process_param_change may be overridden to apply any necessary
    transformations.
    """
    _timeout: ClassVar[int] = 20000
    _debounce: ClassVar[int] = 50
    _priority_changes: ClassVar[List[str]] = []
    _manual_params: ClassVar[List[str]] = []
    _rename: ClassVar[Mapping[str, str | None]] = {}
    _js_transforms: ClassVar[Mapping[str, str]] = {}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {}
    _target_transforms: ClassVar[Mapping[str, str | None]] = {}
    _stylesheets: ClassVar[List[str]] = []
    _busy__ignore = []
    __abstract = True

    def __init__(self, **params):
        self._themer = None
        super().__init__(**params)
        self._updating = False
        self._events = {}
        self._links = []
        self._link_params()
        self._changing = {}
        if self._manual_params:
            self._internal_callbacks.append(self.param.watch(self._update_manual, self._manual_params))

    @classproperty
    @lru_cache(maxsize=None)
    def _property_mapping(cls):
        rename = {}
        for scls in cls.__mro__[::-1]:
            if issubclass(scls, Syncable):
                rename.update(scls._rename)
        return rename

    @property
    def _linked_properties(self) -> Tuple[str]:
        return tuple((self._property_mapping.get(p, p) for p in self.param if p not in Viewable.param and self._property_mapping.get(p, p) is not None))

    def _get_properties(self, doc: Document) -> Dict[str, Any]:
        return self._process_param_change(self._init_params())

    def _process_property_change(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform bokeh model property changes into parameter updates.
        Should be overridden to provide appropriate mapping between
        parameter value and bokeh model change. By default uses the
        _rename class level attribute to map between parameter and
        property names.
        """
        inverted = {v: k for k, v in self._property_mapping.items()}
        return {inverted.get(k, k): v for k, v in msg.items()}

    def _process_param_change(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform parameter changes into bokeh model property updates.
        Should be overridden to provide appropriate mapping between
        parameter value and bokeh model change. By default uses the
        _rename class level attribute to map between parameter and
        property names.
        """
        properties = {self._property_mapping.get(k) or k: v for k, v in msg.items() if self._property_mapping.get(k, False) is not None and k not in self._manual_params}
        if 'width' in properties and self.sizing_mode is None:
            properties['min_width'] = properties['width']
        if 'height' in properties and self.sizing_mode is None:
            properties['min_height'] = properties['height']
        if 'stylesheets' in properties:
            from .config import config
            stylesheets = [loading_css(config.loading_spinner, config.loading_color, config.loading_max_height), f'{CDN_DIST}css/loading.css']
            stylesheets += process_raw_css(config.raw_css)
            stylesheets += config.css_files
            stylesheets += [resolve_stylesheet(self, css_file, '_stylesheets') for css_file in self._stylesheets]
            stylesheets += properties['stylesheets']
            wrapped = []
            for stylesheet in stylesheets:
                if isinstance(stylesheet, str) and stylesheet.endswith('.css'):
                    stylesheet = ImportedStyleSheet(url=stylesheet)
                wrapped.append(stylesheet)
            properties['stylesheets'] = wrapped
        return properties

    @property
    def _linkable_params(self) -> List[str]:
        """
        Parameters that can be linked in JavaScript via source transforms.
        """
        return [p for p in self._synced_params if self._rename.get(p, False) is not None and self._source_transforms.get(p, False) is not None and (p not in ('design', 'stylesheets'))]

    @property
    def _synced_params(self) -> List[str]:
        """
        Parameters which are synced with properties using transforms
        applied in the _process_param_change method.
        """
        ignored = ['default_layout', 'loading', 'background']
        return [p for p in self.param if p not in self._manual_params + ignored]

    def _init_params(self) -> Dict[str, Any]:
        return {k: v for k, v in self.param.values().items() if k in self._synced_params and v is not None}

    def _link_params(self) -> None:
        params = self._synced_params
        if params:
            watcher = self.param.watch(self._param_change, params)
            self._internal_callbacks.append(watcher)

    def _link_props(self, model: Model, properties: List[str] | List[Tuple[str, str]], doc: Document, root: Model, comm: Optional[Comm]=None) -> None:
        from .config import config
        ref = root.ref['id']
        if config.embed:
            return
        for p in properties:
            if isinstance(p, tuple):
                _, p = p
            m = model
            if '.' in p:
                *subpath, p = p.split('.')
                for sp in subpath:
                    m = getattr(m, sp)
            else:
                subpath = None
            if comm:
                m.on_change(p, partial(self._comm_change, doc, ref, comm, subpath))
            else:
                m.on_change(p, partial(self._server_change, doc, ref, subpath))

    def _manual_update(self, events: Tuple[param.parameterized.Event, ...], model: Model, doc: Document, root: Model, parent: Optional[Model], comm: Optional[Comm]) -> None:
        """
        Method for handling any manual update events, i.e. events triggered
        by changes in the manual params.
        """

    def _update_manual(self, *events: param.parameterized.Event) -> None:
        for ref, (model, parent) in self._models.items():
            if ref not in state._views or ref in state._fake_roots:
                continue
            viewable, root, doc, comm = state._views[ref]
            if comm or state._unblocked(doc):
                with unlocked():
                    self._manual_update(events, model, doc, root, parent, comm)
                if comm and 'embedded' not in root.tags:
                    push(doc, comm)
            else:
                cb = partial(self._manual_update, events, model, doc, root, parent, comm)
                if doc.session_context:
                    doc.add_next_tick_callback(cb)
                else:
                    cb()

    def _apply_update(self, events: Dict[str, param.parameterized.Event], msg: Dict[str, Any], model: Model, ref: str) -> None:
        if ref not in state._views or ref in state._fake_roots:
            return
        viewable, root, doc, comm = state._views[ref]
        if comm or not doc.session_context or state._unblocked(doc):
            with unlocked():
                self._update_model(events, msg, root, model, doc, comm)
            if comm and 'embedded' not in root.tags:
                push(doc, comm)
        else:
            cb = partial(self._update_model, events, msg, root, model, doc, comm)
            doc.add_next_tick_callback(cb)

    def _update_model(self, events: Dict[str, param.parameterized.Event], msg: Dict[str, Any], root: Model, model: Model, doc: Document, comm: Optional[Comm]) -> None:
        ref = root.ref['id']
        self._changing[ref] = attrs = []
        for attr, value in msg.items():
            try:
                model_val = getattr(model, attr)
            except UnsetValueError:
                attrs.append(attr)
                continue
            if not model.lookup(attr).property.matches(model_val, value):
                attrs.append(attr)
            if attr in self._events:
                del self._events[attr]
        try:
            model.update(**msg)
        finally:
            changing = [attr for attr in self._changing.get(ref, []) if attr not in attrs]
            if changing:
                self._changing[ref] = changing
            elif ref in self._changing:
                del self._changing[ref]

    def _cleanup(self, root: Model | None) -> None:
        super()._cleanup(root)
        if root is None:
            return
        ref = root.ref['id']
        if ref in self._models:
            model, _ = self._models.pop(ref, None)
            model._callbacks = {}
            model._event_callbacks = {}
        comm, client_comm = self._comms.pop(ref, (None, None))
        if comm:
            try:
                comm.close()
            except Exception:
                pass
        if client_comm:
            try:
                client_comm.close()
            except Exception:
                pass

    def _update_properties(self, *events: param.parameterized.Event, doc: Document) -> Dict[str, Any]:
        changes = {event.name: event.new for event in events}
        return self._process_param_change(changes)

    def _param_change(self, *events: param.parameterized.Event) -> None:
        named_events = {event.name: event for event in events}
        for ref, (model, _) in self._models.copy().items():
            properties = self._update_properties(*events, doc=model.document)
            if not properties:
                return
            self._apply_update(named_events, properties, model, ref)

    def _process_events(self, events: Dict[str, Any]) -> None:
        self._log('received events %s', events)
        if any((e for e in events if e not in self._busy__ignore)):
            with edit_readonly(state):
                state._busy_counter += 1
        params = self._process_property_change(dict(events))
        try:
            with edit_readonly(self):
                self_params = {k: v for k, v in params.items() if '.' not in k}
                with _syncing(self, list(self_params)):
                    self.param.update(**self_params)
            for k, v in params.items():
                if '.' not in k:
                    continue
                *subpath, p = k.split('.')
                obj = self
                for sp in subpath:
                    obj = getattr(obj, sp)
                with edit_readonly(obj):
                    with _syncing(obj, [p]):
                        obj.param.update(**{p: v})
        except Exception:
            if len(params) > 1:
                msg_end = f'changing properties {pformat(params)} \n'
            elif len(params) == 1:
                msg_end = f'changing property {pformat(params)} \n'
            else:
                msg_end = '\n'
            log.exception(f'Callback failed for object named {self.name!r} {msg_end}')
            raise
        finally:
            self._log('finished processing events %s', events)
            if any((e for e in events if e not in self._busy__ignore)):
                with edit_readonly(state):
                    state._busy_counter -= 1

    def _process_bokeh_event(self, doc: Document, event: Event) -> None:
        self._log('received bokeh event %s', event)
        with edit_readonly(state):
            state._busy_counter += 1
        try:
            with set_curdoc(doc):
                self._process_event(event)
        finally:
            self._log('finished processing bokeh event %s', event)
            with edit_readonly(state):
                state._busy_counter -= 1

    async def _change_coroutine(self, doc: Document) -> None:
        if state._thread_pool:
            future = state._thread_pool.submit(self._change_event, doc)
            future.add_done_callback(partial(state._handle_future_exception, doc=doc))
        else:
            with set_curdoc(doc):
                try:
                    self._change_event(doc)
                except Exception as e:
                    state._handle_exception(e)

    async def _event_coroutine(self, doc: Document, event) -> None:
        if state._thread_pool:
            future = state._thread_pool.submit(self._process_bokeh_event, doc, event)
            future.add_done_callback(partial(state._handle_future_exception, doc=doc))
        else:
            try:
                self._process_bokeh_event(doc, event)
            except Exception as e:
                state._handle_exception(e)

    def _change_event(self, doc: Document) -> None:
        events = self._events
        self._events = {}
        with set_curdoc(doc):
            self._process_events(events)

    def _schedule_change(self, doc: Document, comm: Comm | None) -> None:
        with hold(doc, comm=comm):
            self._change_event(doc)

    def _comm_change(self, doc: Document, ref: str, comm: Comm | None, subpath: str, attr: str, old: Any, new: Any) -> None:
        if subpath:
            attr = f'{subpath}.{attr}'
        if attr in self._changing.get(ref, []):
            self._changing[ref].remove(attr)
            return
        self._events.update({attr: new})
        if state._thread_pool:
            future = state._thread_pool.submit(self._schedule_change, doc, comm)
            future.add_done_callback(partial(state._handle_future_exception, doc=doc))
        else:
            try:
                self._schedule_change(doc, comm)
            except Exception as e:
                state._handle_exception(e)

    def _comm_event(self, doc: Document, event: Event) -> None:
        if state._thread_pool:
            future = state._thread_pool.submit(self._process_bokeh_event, doc, event)
            future.add_done_callback(partial(state._handle_future_exception, doc=doc))
        else:
            try:
                self._process_bokeh_event(doc, event)
            except Exception as e:
                state._handle_exception(e)

    def _register_events(self, *event_names: str, model: Model, doc: Document, comm: Comm | None) -> None:
        for event_name in event_names:
            method = self._comm_event if comm else self._server_event
            model.on_event(event_name, partial(method, doc))

    def _server_event(self, doc: Document, event: Event) -> None:
        if doc.session_context and (not state._unblocked(doc)):
            doc.add_next_tick_callback(partial(self._event_coroutine, doc, event))
        else:
            self._comm_event(doc, event)

    def _server_change(self, doc: Document, ref: str, subpath: str, attr: str, old: Any, new: Any) -> None:
        if subpath:
            attr = f'{subpath}.{attr}'
        if attr in self._changing.get(ref, []):
            self._changing[ref].remove(attr)
            return
        processing = bool(self._events)
        self._events.update({attr: new})
        if processing:
            return
        if doc.session_context:
            cb = partial(self._change_coroutine, doc)
            if attr in self._priority_changes:
                doc.add_next_tick_callback(cb)
            else:
                doc.add_timeout_callback(cb, self._debounce)
        else:
            try:
                self._change_event(doc)
            except Exception as e:
                state._handle_exception(e)