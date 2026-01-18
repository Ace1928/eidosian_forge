from a list of options.
from __future__ import annotations
import itertools
import re
from types import FunctionType
from typing import (
import numpy as np
import param
from bokeh.models import PaletteSelect
from bokeh.models.widgets import (
from ..io.resources import CDN_DIST
from ..layout.base import Column, ListPanel, NamedListPanel
from ..models import (
from ..util import PARAM_NAME_PATTERN, indexOf, isIn
from ._mixin import TooltipMixin
from .base import CompositeWidget, Widget
from .button import Button, _ButtonBase
from .input import TextAreaInput, TextInput
class NestedSelect(CompositeWidget):
    """
    The `NestedSelect` widget is composed of multiple widgets, where subsequent select options
    depend on the parent's value.

    Reference: https://panel.holoviz.org/reference/widgets/NestedSelect.html

    :Example:

    >>> NestedSelect(
    ...     options={
    ...         "gfs": {"tmp": [1000, 500], "pcp": [1000]},
    ...         "name": {"tmp": [1000, 925, 850, 700, 500], "pcp": [1000]},
    ...     },
    ...     levels=["model", "var", "level"],
    ... )
    """
    value = param.Dict(doc='\n        The value from all the Select widgets; the keys are the levels names.\n        If no levels names are specified, the keys are the levels indices.')
    options = param.ClassSelector(class_=(dict, FunctionType), doc='\n        The options to select from. The options may be nested dictionaries, lists,\n        or callables that return those types. If callables are used, the callables\n        must accept `level` and `value` keyword arguments, where `level` is the\n        level that updated and `value` is a dictionary of the current values, containing keys\n        up to the level that was updated.')
    layout = param.Parameter(default=Column, doc='\n        The layout type of the widgets. If a dictionary, a "type" key can be provided,\n        to specify the layout type of the widgets, and any additional keyword arguments\n        will be used to instantiate the layout.')
    levels = param.List(doc='\n        Either a list of strings or a list of dictionaries. If a list of strings, the strings\n        are used as the names of the levels. If a list of dictionaries, each dictionary may\n        have a "name" key, which is used as the name of the level, a "type" key, which\n        is used as the type of widget, and any corresponding widget keyword arguments.\n        Must be specified if options is callable.')
    disabled = param.Boolean(default=False, doc='\n        Whether the widget is disabled.')
    _widgets = param.List(doc='The nested select widgets.')
    _max_depth = param.Integer(doc='The number of levels of the nested select widgets.')
    _levels = param.List(doc='\n        The internal rep of levels to prevent overwriting user provided levels.')

    def __init__(self, **params):
        super().__init__(**params)
        self._update_widgets()

    def _gather_values_from_widgets(self, up_to_i=None):
        """
        Gather values from all the select widgets to update the class' value.
        """
        values = {}
        for i, select in enumerate(self._widgets):
            if up_to_i is not None and i >= up_to_i:
                break
            level = self._levels[i]
            if isinstance(level, dict):
                name = level.get('name', i)
            else:
                name = level
            values[name] = select.value if select.options else None
        return values

    def _uses_callable(self, d):
        """
        Check if the nested options has a callable.
        """
        if callable(d):
            return True
        if isinstance(d, dict):
            for value in d.values():
                if callable(value):
                    return True
                elif isinstance(value, dict):
                    return self._uses_callable(value)
        return False

    def _find_max_depth(self, d, depth=1):
        if d is None or len(d) == 0:
            return 0
        elif not isinstance(d, dict):
            return depth
        max_depth = depth
        for value in d.values():
            if isinstance(value, dict):
                max_depth = max(max_depth, self._find_max_depth(value, depth + 1))
            if isinstance(value, list) and len(value) == 0 and (max_depth > 0):
                max_depth -= 1
        return max_depth

    def _resolve_callable_options(self, i, options) -> dict | list:
        level = self.levels[i]
        value = self._gather_values_from_widgets(up_to_i=i)
        options = options(level=level, value=value)
        return options

    @param.depends('options', 'layout', 'levels', watch=True)
    def _update_widgets(self):
        """
        When options is changed, reflect changes on the select widgets.
        """
        if self._uses_callable(self.options):
            if not self.levels:
                raise ValueError('levels must be specified if options is callable')
            self._max_depth = len(self.levels)
        else:
            self._max_depth = self._find_max_depth(self.options) + 1
        if not self.levels:
            self._levels = [i for i in range(self._max_depth)]
        elif len(self.levels) != self._max_depth:
            raise ValueError(f'levels must be of length {self._max_depth}')
        else:
            self._levels = self.levels
        self._widgets = []
        options = self.options or []
        if isinstance(self.options, dict):
            options = self.options.copy()
        for i in range(self._max_depth):
            if callable(options):
                options = self._resolve_callable_options(i, options)
            value = self._init_widget(i, options)
            if isinstance(options, dict) and len(options) > 0 and (value is not None):
                options = options[value]
            elif i < self._max_depth - 1 and (not isinstance(options, dict)):
                raise ValueError(f'The level, {self.levels[i]!r} is not the last nested level, so it must be a dict, but got {options!r}, which is a {type(options).__name__}')
        if isinstance(self.layout, dict):
            layout_type = self.layout.pop('type', Column)
            layout_kwargs = self.layout.copy()
        elif issubclass(self.layout, (ListPanel, NamedListPanel)):
            layout_type = self.layout
            layout_kwargs = {}
        else:
            raise ValueError(f'The layout must be a subclass of ListLike or dict, got {self.layout!r}.')
        self._composite = layout_type(*self._widgets, **layout_kwargs)
        if self.options is not None:
            self.value = self._gather_values_from_widgets()

    def _extract_level_metadata(self, i):
        """
        Extract the widget type and keyword arguments from the level metadata.
        """
        level = self._levels[i]
        if isinstance(level, int):
            return (Select, {})
        elif isinstance(level, str):
            return (Select, {'name': level})
        widget_type = level.get('type', Select)
        widget_kwargs = {k: v for k, v in level.items() if k != 'type'}
        return (widget_type, widget_kwargs)

    def _lookup_value(self, i, options, values, name=None, error=False):
        """
        Look up the value of the select widget at index i or by name.
        """
        options_iterable = isinstance(options, (list, dict))
        if values is None or (options_iterable and len(options) == 0):
            value = None
        elif name is None:
            value = list(values.values())[i] if i < len(values) else None
        elif isinstance(self._levels[0], int):
            value = values.get(i)
        else:
            value = values.get(name)
        if options_iterable and options and (value not in options):
            if value is not None and error:
                raise ValueError(f'Failed to set value {value!r} for level {name!r}, must be one of {options!r}.')
            else:
                value = options[0]
        return value

    def _init_widget(self, i, options):
        """
        Helper method to initialize a select widget.
        """
        if isinstance(options, dict):
            options = list(options.keys())
        elif not isinstance(options, (list, dict)) and (not callable(options)):
            raise ValueError(f'options must be a dict, list, or callable that returns those types, got {options!r}, which is a {type(options).__name__}')
        widget_type, widget_kwargs = self._extract_level_metadata(i)
        value = self._lookup_value(i, options, self.value, error=False)
        widget_kwargs['options'] = options
        widget_kwargs['value'] = value
        if 'visible' not in widget_kwargs:
            widget_kwargs['visible'] = i == 0 or callable(options) or len(options) > 0
        widget = widget_type(**widget_kwargs)
        self.link(widget, disabled='disabled')
        widget.param.watch(self._update_widget_options_interactively, 'value')
        self._widgets.append(widget)
        return value

    def _update_widget_options_interactively(self, event):
        """
        When a select widget's value is changed, update to the latest options.
        """
        if self.options is None:
            return
        for start_i, select in enumerate(self._widgets):
            if select is event.obj:
                break
        options = self.options if callable(self.options) else self.options.copy()
        with param.parameterized.batch_call_watchers(self):
            for i, select in enumerate(self._widgets[:-1]):
                if select.value is None:
                    options = {}
                    visible = False
                elif options:
                    if isinstance(options, dict):
                        if select.value in options:
                            options = options[select.value]
                        else:
                            options = options[list(options.keys())[0]]
                    visible = bool(options)
                if i < start_i:
                    continue
                next_select = self._widgets[i + 1]
                if callable(options):
                    options = self._resolve_callable_options(i + 1, options)
                    next_options = list(options)
                elif isinstance(options, dict):
                    next_options = list(options.keys())
                elif isinstance(options, list):
                    next_options = options
                else:
                    raise NotImplementedError('options must be a dict, list, or callable that returns those types.')
                next_select.param.update(options=next_options, visible=visible)
            self.value = self._gather_values_from_widgets()

    @param.depends('value', watch=True)
    def _update_options_programmatically(self):
        """
        When value is passed, update to the latest options.
        """
        if self.options is None:
            return
        options = self.options if callable(self.options) else self.options.copy()
        set_values = self.value.copy()
        original_values = self._gather_values_from_widgets()
        if set_values == original_values:
            return
        with param.parameterized.batch_call_watchers(self):
            try:
                for i in range(self._max_depth):
                    curr_select = self._widgets[i]
                    if callable(options):
                        options = self._resolve_callable_options(i, options)
                        curr_options = list(options)
                    elif isinstance(options, dict):
                        curr_options = list(options.keys())
                    else:
                        curr_options = options
                    curr_value = self._lookup_value(i, curr_options, set_values, name=curr_select.name, error=True)
                    with param.discard_events(self):
                        curr_select.param.update(options=curr_options, value=curr_value, visible=callable(curr_options) or len(curr_options) > 0)
                    if curr_value is None:
                        break
                    if i < self._max_depth - 1:
                        options = options[curr_value]
            except Exception:
                self.value = original_values
                raise