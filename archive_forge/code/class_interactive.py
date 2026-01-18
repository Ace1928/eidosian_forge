from collections.abc import Iterable, Mapping
from inspect import signature, Parameter
from inspect import getcallargs
from inspect import getfullargspec as check_argspec
import sys
from IPython import get_ipython
from . import (Widget, ValueWidget, Text,
from IPython.display import display, clear_output
from traitlets import HasTraits, Any, Unicode, observe
from numbers import Real, Integral
from warnings import warn
class interactive(VBox):
    """
    A VBox container containing a group of interactive widgets tied to a
    function.

    Parameters
    ----------
    __interact_f : function
        The function to which the interactive widgets are tied. The `**kwargs`
        should match the function signature.
    __options : dict
        A dict of options. Currently, the only supported keys are
        ``"manual"`` (defaults to ``False``), ``"manual_name"`` (defaults
        to ``"Run Interact"``) and ``"auto_display"`` (defaults to ``False``).
    **kwargs : various, optional
        An interactive widget is created for each keyword argument that is a
        valid widget abbreviation.

    Note that the first two parameters intentionally start with a double
    underscore to avoid being mixed up with keyword arguments passed by
    ``**kwargs``.
    """

    def __init__(self, __interact_f, __options={}, **kwargs):
        VBox.__init__(self, _dom_classes=['widget-interact'])
        self.result = None
        self.args = []
        self.kwargs = {}
        self.f = f = __interact_f
        self.clear_output = kwargs.pop('clear_output', True)
        self.manual = __options.get('manual', False)
        self.manual_name = __options.get('manual_name', 'Run Interact')
        self.auto_display = __options.get('auto_display', False)
        new_kwargs = self.find_abbreviations(kwargs)
        try:
            check_argspec(f)
        except TypeError:
            pass
        else:
            getcallargs(f, **{n: v for n, v, _ in new_kwargs})
        self.kwargs_widgets = self.widgets_from_abbreviations(new_kwargs)
        c = [w for w in self.kwargs_widgets if isinstance(w, DOMWidget)]
        if self.manual:
            self.manual_button = Button(description=self.manual_name)
            c.append(self.manual_button)
        self.out = Output()
        c.append(self.out)
        self.children = c
        if self.manual:
            self.manual_button.on_click(self.update)
            for w in self.kwargs_widgets:
                if isinstance(w, Text):
                    w.continuous_update = False
                    w.observe(self.update, names='value')
        else:
            for widget in self.kwargs_widgets:
                widget.observe(self.update, names='value')
            self.update()

    def update(self, *args):
        """
        Call the interact function and update the output widget with
        the result of the function call.

        Parameters
        ----------
        *args : ignored
            Required for this method to be used as traitlets callback.
        """
        self.kwargs = {}
        if self.manual:
            self.manual_button.disabled = True
        try:
            show_inline_matplotlib_plots()
            with self.out:
                if self.clear_output:
                    clear_output(wait=True)
                for widget in self.kwargs_widgets:
                    value = widget.get_interact_value()
                    self.kwargs[widget._kwarg] = value
                self.result = self.f(**self.kwargs)
                show_inline_matplotlib_plots()
                if self.auto_display and self.result is not None:
                    display(self.result)
        except Exception as e:
            ip = get_ipython()
            if ip is None:
                self.log.warning('Exception in interact callback: %s', e, exc_info=True)
            else:
                ip.showtraceback()
        finally:
            if self.manual:
                self.manual_button.disabled = False

    def signature(self):
        return signature(self.f)

    def find_abbreviations(self, kwargs):
        """Find the abbreviations for the given function and kwargs.
        Return (name, abbrev, default) tuples.
        """
        new_kwargs = []
        try:
            sig = self.signature()
        except (ValueError, TypeError):
            return [(key, value, value) for key, value in kwargs.items()]
        for param in sig.parameters.values():
            for name, value, default in _yield_abbreviations_for_parameter(param, kwargs):
                if value is empty:
                    raise ValueError('cannot find widget or abbreviation for argument: {!r}'.format(name))
                new_kwargs.append((name, value, default))
        return new_kwargs

    def widgets_from_abbreviations(self, seq):
        """Given a sequence of (name, abbrev, default) tuples, return a sequence of Widgets."""
        result = []
        for name, abbrev, default in seq:
            if isinstance(abbrev, Widget) and (not isinstance(abbrev, ValueWidget)):
                raise TypeError('{!r} is not a ValueWidget'.format(abbrev))
            widget = self.widget_from_abbrev(abbrev, default)
            if widget is None:
                raise ValueError('{!r} cannot be transformed to a widget'.format(abbrev))
            if not hasattr(widget, 'description') or not widget.description:
                widget.description = name
            widget._kwarg = name
            result.append(widget)
        return result

    @classmethod
    def widget_from_abbrev(cls, abbrev, default=empty):
        """Build a ValueWidget instance given an abbreviation or Widget."""
        if isinstance(abbrev, ValueWidget) or isinstance(abbrev, fixed):
            return abbrev
        if isinstance(abbrev, tuple):
            widget = cls.widget_from_tuple(abbrev)
            if default is not empty:
                try:
                    widget.value = default
                except Exception:
                    pass
            return widget
        widget = cls.widget_from_single_value(abbrev)
        if widget is not None:
            return widget
        if isinstance(abbrev, Iterable):
            widget = cls.widget_from_iterable(abbrev)
            if default is not empty:
                try:
                    widget.value = default
                except Exception:
                    pass
            return widget
        return None

    @staticmethod
    def widget_from_single_value(o):
        """Make widgets from single values, which can be used as parameter defaults."""
        if isinstance(o, str):
            return Text(value=str(o))
        elif isinstance(o, bool):
            return Checkbox(value=o)
        elif isinstance(o, Integral):
            min, max, value = _get_min_max_value(None, None, o)
            return IntSlider(value=o, min=min, max=max)
        elif isinstance(o, Real):
            min, max, value = _get_min_max_value(None, None, o)
            return FloatSlider(value=o, min=min, max=max)
        else:
            return None

    @staticmethod
    def widget_from_tuple(o):
        """Make widgets from a tuple abbreviation."""
        if _matches(o, (Real, Real)):
            min, max, value = _get_min_max_value(o[0], o[1])
            if all((isinstance(_, Integral) for _ in o)):
                cls = IntSlider
            else:
                cls = FloatSlider
            return cls(value=value, min=min, max=max)
        elif _matches(o, (Real, Real, Real)):
            step = o[2]
            if step <= 0:
                raise ValueError('step must be >= 0, not %r' % step)
            min, max, value = _get_min_max_value(o[0], o[1], step=step)
            if all((isinstance(_, Integral) for _ in o)):
                cls = IntSlider
            else:
                cls = FloatSlider
            return cls(value=value, min=min, max=max, step=step)

    @staticmethod
    def widget_from_iterable(o):
        """Make widgets from an iterable. This should not be done for
        a string or tuple."""
        if isinstance(o, (list, dict)):
            return Dropdown(options=o)
        elif isinstance(o, Mapping):
            return Dropdown(options=list(o.items()))
        else:
            return Dropdown(options=list(o))

    @classmethod
    def factory(cls):
        options = dict(manual=False, auto_display=True, manual_name='Run Interact')
        return _InteractFactory(cls, options)