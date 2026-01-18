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
class Reactive(Syncable, Viewable):
    """
    Reactive is a Viewable object that also supports syncing between
    the objects parameters and the underlying bokeh model either via
    the defined pyviz_comms.Comm type or using bokeh server.

    In addition it defines various methods which make it easy to link
    the parameters to other objects.
    """
    _ignored_refs: ClassVar[Tuple[str, ...]] = ()
    _rename: ClassVar[Mapping[str, str | None]] = {'design': None, 'loading': None}
    __abstract = True

    def __init__(self, refs=None, **params):
        for name, pobj in self.param.objects('existing').items():
            if name not in self._param__private.explicit_no_refs:
                pobj.allow_refs = True
        if refs is not None:
            self._refs = refs
            if iscoroutinefunction(refs):
                param.parameterized.async_executor(self._async_refs)
            else:
                params.update(resolve_value(self._refs))
            refs = resolve_ref(self._refs)
            if refs:
                param.bind(self._sync_refs, *refs, watch=True)
        super().__init__(**params)

    def _sync_refs(self, *_):
        resolved = resolve_value(self._refs)
        self.param.update(resolved)

    async def _async_refs(self, *_):
        resolved = resolve_value(self._refs)
        if inspect.isasyncgenfunction(self._refs):
            async for val in resolved:
                self.param.update(val)
        else:
            self.param.update(await resolved)

    def _get_properties(self, doc: Document) -> Dict[str, Any]:
        params, _ = self._design.params(self, doc) if self._design else ({}, None)
        for k, v in self._init_params().items():
            if k in ('stylesheets', 'tags') and k in params:
                params[k] = v = params[k] + v
            elif k not in params or self.param[k].default is not v:
                params[k] = v
        properties = self._process_param_change(params)
        if 'stylesheets' not in properties:
            return properties
        if doc:
            state._stylesheets[doc] = cache = state._stylesheets.get(doc, {})
        else:
            cache = {}
        if doc and 'dist_url' in doc._template_variables:
            dist_url = doc._template_variables['dist_url']
        else:
            dist_url = CDN_DIST
        stylesheets = []
        for stylesheet in properties['stylesheets']:
            if isinstance(stylesheet, ImportedStyleSheet):
                if stylesheet.url in cache:
                    stylesheet = cache[stylesheet.url]
                else:
                    cache[stylesheet.url] = stylesheet
                patch_stylesheet(stylesheet, dist_url)
            stylesheets.append(stylesheet)
        properties['stylesheets'] = stylesheets
        return properties

    def _update_properties(self, *events: param.parameterized.Event, doc: Document) -> Dict[str, Any]:
        params, _ = self._design.params(self, doc) if self._design else ({}, None)
        changes = {event.name: event.new for event in events}
        if 'stylesheets' in changes and 'stylesheets' in params:
            changes['stylesheets'] = params['stylesheets'] + changes['stylesheets']
        return self._process_param_change(changes)

    def _update_model(self, events: Dict[str, param.parameterized.Event], msg: Dict[str, Any], root: Model, model: Model, doc: Document, comm: Optional[Comm]) -> None:
        if 'stylesheets' in msg:
            if doc and 'dist_url' in doc._template_variables:
                dist_url = doc._template_variables['dist_url']
            else:
                dist_url = CDN_DIST
            for stylesheet in msg['stylesheets']:
                if isinstance(stylesheet, ImportedStyleSheet):
                    patch_stylesheet(stylesheet, dist_url)
        super()._update_model(events, msg, root, model, doc, comm)

    def link(self, target: param.Parameterized, callbacks: Optional[Dict[str, str | Callable]]=None, bidirectional: bool=False, **links: str) -> Watcher:
        """
        Links the parameters on this `Reactive` object to attributes on the
        target `Parameterized` object.

        Supports two modes, either specify a
        mapping between the source and target object parameters as keywords or
        provide a dictionary of callbacks which maps from the source
        parameter to a callback which is triggered when the parameter
        changes.

        Arguments
        ---------
        target: param.Parameterized
          The target object of the link.
        callbacks: dict | None
          Maps from a parameter in the source object to a callback.
        bidirectional: bool
          Whether to link source and target bi-directionally
        **links: dict
          Maps between parameters on this object to the parameters
          on the supplied object.
        """
        if links and callbacks:
            raise ValueError('Either supply a set of parameters to link as keywords or a set of callbacks, not both.')
        elif not links and (not callbacks):
            raise ValueError('Declare parameters to link or a set of callbacks, neither was defined.')
        elif callbacks and bidirectional:
            raise ValueError('Bidirectional linking not supported for explicit callbacks. You must define separate callbacks for each direction.')
        _updating = []

        def link_cb(*events):
            for event in events:
                if event.name in _updating:
                    continue
                _updating.append(event.name)
                try:
                    if callbacks:
                        callbacks[event.name](target, event)
                    else:
                        setattr(target, links[event.name], event.new)
                finally:
                    _updating.pop(_updating.index(event.name))
        params = list(callbacks) if callbacks else list(links)
        cb = self.param.watch(link_cb, params)
        bidirectional_watcher = None
        if bidirectional:
            _reverse_updating = []
            reverse_links = {v: k for k, v in links.items()}

            def reverse_link(*events):
                for event in events:
                    if event.name in _reverse_updating:
                        continue
                    _reverse_updating.append(event.name)
                    try:
                        setattr(self, reverse_links[event.name], event.new)
                    finally:
                        _reverse_updating.remove(event.name)
            bidirectional_watcher = target.param.watch(reverse_link, list(reverse_links))
        link_args = tuple(cb)
        if 'precedence' in Watcher._fields and len(link_args) < len(Watcher._fields):
            link_args += (cb.precedence,)
        link = LinkWatcher(*link_args + (target, links, callbacks is not None, bidirectional_watcher))
        self._links.append(link)
        return cb

    def controls(self, parameters: List[str]=[], jslink: bool=True, **kwargs) -> 'Panel':
        """
        Creates a set of widgets which allow manipulating the parameters
        on this instance. By default all parameters which support
        linking are exposed, but an explicit list of parameters can
        be provided.

        Arguments
        ---------
        parameters: list(str)
           An explicit list of parameters to return controls for.
        jslink: bool
           Whether to use jslinks instead of Python based links.
           This does not allow using all types of parameters.
        kwargs: dict
           Additional kwargs to pass to the Param pane(s) used to
           generate the controls widgets.

        Returns
        -------
        A layout of the controls
        """
        from .layout import Tabs, WidgetBox
        from .param import Param
        from .widgets import LiteralInput
        if parameters:
            linkable = parameters
        elif jslink:
            linkable = self._linkable_params
        else:
            linkable = list(self.param)
        params = [p for p in linkable if p not in Viewable.param]
        controls = Param(self.param, parameters=params, default_layout=WidgetBox, name='Controls', **kwargs)
        layout_params = [p for p in linkable if p in Viewable.param]
        if 'name' not in layout_params and self._property_mapping.get('name', False) is not None and (not parameters):
            layout_params.insert(0, 'name')
        style = Param(self.param, parameters=layout_params, default_layout=WidgetBox, name='Layout', **kwargs)
        if jslink:
            for p in params:
                widget = controls._widgets[p]
                widget.jslink(self, value=p, bidirectional=True)
                if isinstance(widget, LiteralInput):
                    widget.serializer = 'json'
            for p in layout_params:
                widget = style._widgets[p]
                widget.jslink(self, value=p, bidirectional=p != 'loading')
                if isinstance(widget, LiteralInput):
                    widget.serializer = 'json'
        if params and layout_params:
            return Tabs(controls.layout[0], style.layout[0])
        elif params:
            return controls.layout[0]
        return style.layout[0]

    def jscallback(self, args: Dict[str, Any]={}, **callbacks: str) -> Callback:
        """
        Allows defining a JS callback to be triggered when a property
        changes on the source object. The keyword arguments define the
        properties that trigger a callback and the JS code that gets
        executed.

        Arguments
        ----------
        args: dict
          A mapping of objects to make available to the JS callback
        **callbacks: dict
          A mapping between properties on the source model and the code
          to execute when that property changes

        Returns
        -------
        callback: Callback
          The Callback which can be used to disable the callback.
        """
        from .links import Callback
        return Callback(self, code=callbacks, args=args)

    def jslink(self, target: JSLinkTarget, code: Dict[str, str]=None, args: Optional[Dict]=None, bidirectional: bool=False, **links: str) -> Link:
        """
        Links properties on the this Reactive object to those on the
        target Reactive object in JS code.

        Supports two modes, either specify a
        mapping between the source and target model properties as
        keywords or provide a dictionary of JS code snippets which
        maps from the source parameter to a JS code snippet which is
        executed when the property changes.

        Arguments
        ----------
        target: panel.viewable.Viewable | bokeh.model.Model | holoviews.core.dimension.Dimensioned
          The target to link the value to.
        code: dict
          Custom code which will be executed when the widget value
          changes.
        args: dict
          A mapping of objects to make available to the JS callback
        bidirectional: boolean
          Whether to link source and target bi-directionally
        **links: dict
          A mapping between properties on the source model and the
          target model property to link it to.

        Returns
        -------
        link: GenericLink
          The GenericLink which can be used unlink the widget and
          the target model.
        """
        if links and code:
            raise ValueError('Either supply a set of properties to link as keywords or a set of JS code callbacks, not both.')
        elif not links and (not code):
            raise ValueError('Declare parameters to link or a set of callbacks, neither was defined.')
        if args is None:
            args = {}
        from .links import Link, assert_source_syncable, assert_target_syncable
        mapping = code or links
        assert_source_syncable(self, mapping)
        if isinstance(target, Syncable) and code is None:
            assert_target_syncable(self, target, mapping)
        return Link(self, target, properties=links, code=code, args=args, bidirectional=bidirectional)