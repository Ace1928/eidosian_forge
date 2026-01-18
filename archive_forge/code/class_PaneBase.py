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
class PaneBase(Reactive):
    """
    PaneBase is the abstract baseclass for all atomic displayable units
    in the Panel library. We call any child class of `PaneBase` a `Pane`.

    Panes defines an extensible interface for wrapping arbitrary
    objects and transforming them into Bokeh models.

    Panes are reactive in the sense that when the object they are
    wrapping is replaced or modified the `bokeh.model.Model` that
    is rendered should reflect these changes.
    """
    default_layout = param.ClassSelector(default=Row, class_=Panel, is_instance=False, doc='\n        Defines the layout the model(s) returned by the pane will\n        be placed in.')
    margin = Margin(default=(5, 10), doc='\n        Allows to create additional space around the component. May\n        be specified as a two-tuple of the form (vertical, horizontal)\n        or a four-tuple (top, right, bottom, left).')
    object = param.Parameter(default=None, allow_refs=True, doc='\n        The object being wrapped, which will be converted to a\n        Bokeh model.')
    priority: ClassVar[float | bool | None] = 0.5
    _applies_kw: ClassVar[bool] = False
    _unpack: ClassVar[bool] = True
    _updates: ClassVar[bool] = False
    _rename: ClassVar[Mapping[str, str | None]] = {'default_layout': None, 'loading': None}
    _rerender_params: ClassVar[List[str]] = ['object']
    _skip_layoutable = ('css_classes', 'margin', 'name')
    __abstract = True

    def __init__(self, object=None, **params):
        self._object_changing = False
        super().__init__(object=object, **params)
        applies = self.applies(self.object, **params if self._applies_kw else {})
        if (isinstance(applies, bool) and (not applies)) and self.object is not None:
            self._type_error(self.object)
        kwargs = {k: v for k, v in params.items() if k in Layoutable.param and k not in self._skip_layoutable}
        self.layout = self.default_layout(self, **kwargs)
        self._internal_callbacks.extend([self.param.watch(self._sync_layoutable, list(Layoutable.param)), self.param.watch(self._update_pane, self._rerender_params)])
        self._sync_layoutable()

    def _validate_ref(self, pname, value):
        super()._validate_ref(pname, value)
        if pname == 'object' and (not self._applies_kw):
            applies = self.applies(value)
            if isinstance(applies, bool) and (not applies):
                raise RuntimeError('Value is not valid.')

    def _sync_layoutable(self, *events: param.parameterized.Event):
        included = set(Layoutable.param) - set(self._skip_layoutable)
        if events:
            kwargs = {event.name: event.new for event in events if event.name in included}
        else:
            kwargs = {k: v for k, v in self.param.values().items() if k in included}
        if self.margin:
            margin = self.margin
            if isinstance(margin, tuple):
                if len(margin) == 2:
                    t = b = margin[0]
                    r = l = margin[1]
                else:
                    t, r, b, l = margin
            else:
                t = r = b = l = margin
            if kwargs.get('width') is not None:
                kwargs['width'] = kwargs['width'] + l + r
            if kwargs.get('height') is not None:
                kwargs['height'] = kwargs['height'] + t + b
        old_values = self.layout.param.values()
        self.layout.param.update({k: v for k, v in kwargs.items() if v != old_values[k]})

    def _type_error(self, object):
        raise ValueError("%s pane does not support objects of type '%s'." % (type(self).__name__, type(object).__name__))

    def __repr__(self, depth: int=0) -> str:
        cls = type(self).__name__
        params = param_reprs(self, ['object'])
        obj = 'None' if self.object is None else type(self.object).__name__
        template = '{cls}({obj}, {params})' if params else '{cls}({obj})'
        return template.format(cls=cls, params=', '.join(params), obj=obj)

    def __getitem__(self, index: int | str) -> Viewable:
        """
        Allows pane objects to behave like the underlying layout
        """
        return self.layout[index]

    @property
    def _linked_properties(self) -> Tuple[str]:
        return tuple((self._property_mapping.get(p, p) for p in self.param if p not in PaneBase.param and self._property_mapping.get(p, p) is not None))

    @property
    def _linkable_params(self) -> List[str]:
        return [p for p in self._synced_params if self._property_mapping.get(p, False) is not None]

    @property
    def _synced_params(self) -> List[str]:
        ignored_params = ['name', 'default_layout', 'loading', 'stylesheets'] + self._rerender_params
        return [p for p in self.param if p not in ignored_params and (not p.startswith('_'))]

    def _param_change(self, *events: param.parameterized.Event) -> None:
        if self._object_changing:
            return
        super()._param_change(*events)

    def _update_object(self, ref: str, doc: 'Document', root: Model, parent: Model, comm: Comm | None) -> None:
        old_model = self._models[ref][0]
        if self._updates:
            self._update(ref, old_model)
            return
        new_model = self._get_model(doc, root, parent, comm)
        try:
            if isinstance(parent, _BkGridBox):
                indexes = [i for i, child in enumerate(parent.children) if child[0] is old_model]
                if indexes:
                    index = indexes[0]
                    new_model = (new_model,) + parent.children[index][1:]
                    parent.children[index] = new_model
                else:
                    raise ValueError
            elif isinstance(parent, _BkReactiveHTML):
                for node, children in parent.children.items():
                    if old_model in children:
                        index = children.index(old_model)
                        new_models = list(children)
                        new_models[index] = new_model
                        parent.children[node] = new_models
                        break
            elif isinstance(parent, _BkTabs):
                index = [tab.child for tab in parent.tabs].index(old_model)
                old_tab = parent.tabs[index]
                props = dict(old_tab.properties_with_values(), child=new_model)
                parent.tabs[index] = _BkTabPanel(**props)
            else:
                index = parent.children.index(old_model)
                parent.children[index] = new_model
        except ValueError:
            self.param.warning(f'{type(self).__name__} pane model {old_model!r} could not be replaced with new model {new_model!r}, ensure that the parent is not modified at the same time the panel is being updated.')
            return
        layout_parent = self.layout._models.get(ref, [None])[0]
        if parent is layout_parent:
            parent.update(**self.layout._compute_sizing_mode(parent.children, dict(sizing_mode=self.layout.sizing_mode, styles=self.layout.styles, width=self.layout.width, min_width=self.layout.min_width, margin=self.layout.margin)))
        from ..io import state
        ref = root.ref['id']
        if ref in state._views:
            state._views[ref][0]._preprocess(root)

    def _update_pane(self, *events) -> None:
        for ref, (_, parent) in self._models.items():
            if ref not in state._views or ref in state._fake_roots:
                continue
            viewable, root, doc, comm = state._views[ref]
            if comm or state._unblocked(doc):
                with unlocked():
                    self._update_object(ref, doc, root, parent, comm)
                if comm and 'embedded' not in root.tags:
                    push(doc, comm)
            else:
                cb = partial(self._update_object, ref, doc, root, parent, comm)
                if doc.session_context:
                    doc.add_next_tick_callback(cb)
                else:
                    cb()

    def _update(self, ref: str, model: Model) -> None:
        """
        If _updates=True this method is used to update an existing
        Bokeh model instead of replacing the model entirely. The
        supplied model should be updated with the current state.
        """
        raise NotImplementedError

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

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        """
        Returns boolean or float indicating whether the Pane
        can render the object.

        If the priority of the pane is set to
        `None`, this method may also be used to define a float priority
        depending on the object being rendered.
        """
        return None

    def clone(self: T, object: Optional[Any]=None, **params) -> T:
        """
        Makes a copy of the Pane sharing the same parameters.

        Arguments
        ---------
        object: Optional new object to render
        params: Keyword arguments override the parameters on the clone.

        Returns
        -------
        Cloned Pane object
        """
        inherited = get_params_to_inherit(self)
        params = dict(inherited, **params)
        old_object = params.pop('object', None)
        if object is None:
            object = old_object
        return type(self)(object, **params)

    def get_root(self, doc: Optional[Document]=None, comm: Comm | None=None, preprocess: bool=True) -> Model:
        """
        Returns the root model and applies pre-processing hooks

        Arguments
        ---------
        doc: bokeh.document.Document
          Optional Bokeh document the bokeh model will be attached to.
        comm: pyviz_comms.Comm
          Optional pyviz_comms when working in notebook
        preprocess: bool (default=True)
          Whether to run preprocessing hooks

        Returns
        -------
        Returns the bokeh model corresponding to this panel object
        """
        doc = create_doc_if_none_exists(doc)
        if self._design and comm:
            wrapper = self._design._wrapper(self)
            if wrapper is self:
                root_view, root = self._get_root_model(doc, comm, preprocess)
            else:
                root_view = wrapper
                root = wrapper.get_root(doc, comm, preprocess)
        else:
            root_view, root = self._get_root_model(doc, comm, preprocess)
        ref = root.ref['id']
        state._views[ref] = (root_view, root, doc, comm)
        return root

    @classmethod
    def get_pane_type(cls, obj: Any, **kwargs) -> Type['PaneBase']:
        """
        Returns the applicable Pane type given an object by resolving
        the precedence of all types whose applies method declares that
        the object is supported.

        Arguments
        ---------
        obj (object): The object type to return a Pane type for

        Returns
        -------
        The applicable Pane type with the highest precedence.
        """
        if isinstance(obj, Viewable):
            return type(obj)
        descendents = []
        for p in param.concrete_descendents(PaneBase).values():
            if p.priority is None:
                applies = True
                try:
                    priority = p.applies(obj, **kwargs if p._applies_kw else {})
                except Exception:
                    priority = False
            else:
                applies = None
                priority = p.priority
            if isinstance(priority, bool) and priority:
                raise ValueError('If a Pane declares no priority the applies method should return a priority value specific to the object type or False, but the %s pane declares no priority.' % p.__name__)
            elif priority is None or priority is False:
                continue
            descendents.append((priority, applies, p))
        pane_types = reversed(sorted(descendents, key=lambda x: x[0]))
        for _, applies, pane_type in pane_types:
            if applies is None:
                try:
                    applies = pane_type.applies(obj, **kwargs if pane_type._applies_kw else {})
                except Exception:
                    applies = False
            if not applies:
                continue
            return pane_type
        raise TypeError('%s type could not be rendered.' % type(obj).__name__)