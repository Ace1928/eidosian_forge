from __future__ import annotations
from collections.abc import Iterable, Mapping
from inspect import Parameter
from numbers import Integral, Number, Real
from typing import Any, Optional, Tuple
import param
from .base import Widget
from .input import Checkbox, TextInput
from .select import Select
from .slider import DiscreteSlider, FloatSlider, IntSlider
class widget(param.ParameterizedFunction):
    """
    Attempts to find a widget appropriate for a given value.

    Arguments
    ---------
    name: str
        The name of the resulting widget.
    value: Any
        The value to deduce a widget from.
    default: Any
        The default value for the resulting widget.
    **params: Any
        Additional keyword arguments to pass to the widget.

    Returns
    -------
    Widget
    """

    def __call__(self, value: Any, name: str, default=empty, **params):
        """Build a ValueWidget instance given an abbreviation or Widget."""
        if isinstance(value, Widget):
            widget = value
        elif isinstance(value, tuple):
            widget = self.widget_from_tuple(value, name, default)
            if default is not empty:
                try:
                    widget.value = default
                except Exception:
                    pass
        else:
            widget = self.widget_from_single_value(value, name)
            if widget is None and isinstance(value, Iterable):
                widget = self.widget_from_iterable(value, name)
                if default is not empty:
                    try:
                        widget.value = default
                    except Exception:
                        pass
        if widget is None:
            widget = fixed(value)
        if params:
            widget.param.update(**params)
        return widget

    @staticmethod
    def widget_from_single_value(o, name):
        """Make widgets from single values, which can be used as parameter defaults."""
        if isinstance(o, str):
            return TextInput(value=str(o), name=name)
        elif isinstance(o, bool):
            return Checkbox(value=o, name=name)
        elif isinstance(o, Integral):
            min, max, value = _get_min_max_value(None, None, o)
            return IntSlider(value=o, start=min, end=max, name=name)
        elif isinstance(o, Real):
            min, max, value = _get_min_max_value(None, None, o)
            return FloatSlider(value=o, start=min, end=max, name=name)
        else:
            return None

    @staticmethod
    def widget_from_tuple(o, name, default=empty):
        """Make widgets from a tuple abbreviation."""
        int_default = default is empty or isinstance(default, int)
        if _matches(o, (Real, Real)):
            min, max, value = _get_min_max_value(o[0], o[1])
            if all((isinstance(_, Integral) for _ in o)) and int_default:
                cls = IntSlider
            else:
                cls = FloatSlider
            return cls(value=value, start=min, end=max, name=name)
        elif _matches(o, (Real, Real, Real)):
            step = o[2]
            if step <= 0:
                raise ValueError('step must be >= 0, not %r' % step)
            min, max, value = _get_min_max_value(o[0], o[1], step=step)
            if all((isinstance(_, Integral) for _ in o)) and int_default:
                cls = IntSlider
            else:
                cls = FloatSlider
            return cls(value=value, start=min, end=max, step=step, name=name)
        elif _matches(o, (Real, Real, Real, Real)):
            step = o[2]
            if step <= 0:
                raise ValueError('step must be >= 0, not %r' % step)
            min, max, value = _get_min_max_value(o[0], o[1], value=o[3], step=step)
            if all((isinstance(_, Integral) for _ in o)):
                cls = IntSlider
            else:
                cls = FloatSlider
            return cls(value=value, start=min, end=max, step=step, name=name)
        elif len(o) == 4:
            min, max, value = _get_min_max_value(o[0], o[1], value=o[3])
            if all((isinstance(_, Integral) for _ in [o[0], o[1], o[3]])):
                cls = IntSlider
            else:
                cls = FloatSlider
            return cls(value=value, start=min, end=max, name=name)

    @staticmethod
    def widget_from_iterable(o, name):
        """Make widgets from an iterable. This should not be done for
        a string or tuple."""
        values = list(o.values()) if isinstance(o, Mapping) else list(o)
        widget_type = DiscreteSlider if all((param._is_number(v) for v in values)) else Select
        if isinstance(o, (list, dict)):
            return widget_type(options=o, name=name)
        elif isinstance(o, Mapping):
            return widget_type(options=list(o.items()), name=name)
        else:
            return widget_type(options=list(o), name=name)