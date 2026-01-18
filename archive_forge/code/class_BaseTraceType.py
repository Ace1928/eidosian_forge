import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
class BaseTraceType(BaseTraceHierarchyType):
    """
    Base class for the all trace types.

    Specific trace type classes (Scatter, Bar, etc.) are code generated as
    subclasses of this class.
    """

    def __init__(self, plotly_name, **kwargs):
        super(BaseTraceHierarchyType, self).__init__(plotly_name, **kwargs)
        self._hover_callbacks = []
        self._unhover_callbacks = []
        self._click_callbacks = []
        self._select_callbacks = []
        self._deselect_callbacks = []
        self._trace_ind = None

    @property
    def uid(self):
        raise NotImplementedError

    @uid.setter
    def uid(self, val):
        raise NotImplementedError

    def on_hover(self, callback, append=False):
        """
        Register function to be called when the user hovers over one or more
        points in this trace

        Note: Callbacks will only be triggered when the trace belongs to a
        instance of plotly.graph_objs.FigureWidget and it is displayed in an
        ipywidget context. Callbacks will not be triggered on figures
        that are displayed using plot/iplot.

        Parameters
        ----------
        callback
            Callable function that accepts 3 arguments

            - this trace
            - plotly.callbacks.Points object
            - plotly.callbacks.InputDeviceState object

        append : bool
            If False (the default), this callback replaces any previously
            defined on_hover callbacks for this trace. If True,
            this callback is appended to the list of any previously defined
            callbacks.

        Returns
        -------
        None

        Examples
        --------

        >>> import plotly.graph_objects as go
        >>> from plotly.callbacks import Points, InputDeviceState
        >>> points, state = Points(), InputDeviceState()

        >>> def hover_fn(trace, points, state):
        ...     inds = points.point_inds
        ...     # Do something

        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])
        >>> trace.on_hover(hover_fn)

        Note: The creation of the `points` and `state` objects is optional,
        it's simply a convenience to help the text editor perform completion
        on the arguments inside `hover_fn`
        """
        if not append:
            del self._hover_callbacks[:]
        if callback:
            self._hover_callbacks.append(callback)

    def _dispatch_on_hover(self, points, state):
        """
        Dispatch points and device state all all hover callbacks
        """
        for callback in self._hover_callbacks:
            callback(self, points, state)

    def on_unhover(self, callback, append=False):
        """
        Register function to be called when the user unhovers away from one
        or more points in this trace.

        Note: Callbacks will only be triggered when the trace belongs to a
        instance of plotly.graph_objs.FigureWidget and it is displayed in an
        ipywidget context. Callbacks will not be triggered on figures
        that are displayed using plot/iplot.

        Parameters
        ----------
        callback
            Callable function that accepts 3 arguments

            - this trace
            - plotly.callbacks.Points object
            - plotly.callbacks.InputDeviceState object

        append : bool
            If False (the default), this callback replaces any previously
            defined on_unhover callbacks for this trace. If True,
            this callback is appended to the list of any previously defined
            callbacks.

        Returns
        -------
        None

        Examples
        --------

        >>> import plotly.graph_objects as go
        >>> from plotly.callbacks import Points, InputDeviceState
        >>> points, state = Points(), InputDeviceState()

        >>> def unhover_fn(trace, points, state):
        ...     inds = points.point_inds
        ...     # Do something

        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])
        >>> trace.on_unhover(unhover_fn)

        Note: The creation of the `points` and `state` objects is optional,
        it's simply a convenience to help the text editor perform completion
        on the arguments inside `unhover_fn`
        """
        if not append:
            del self._unhover_callbacks[:]
        if callback:
            self._unhover_callbacks.append(callback)

    def _dispatch_on_unhover(self, points, state):
        """
        Dispatch points and device state all all hover callbacks
        """
        for callback in self._unhover_callbacks:
            callback(self, points, state)

    def on_click(self, callback, append=False):
        """
        Register function to be called when the user clicks on one or more
        points in this trace.

        Note: Callbacks will only be triggered when the trace belongs to a
        instance of plotly.graph_objs.FigureWidget and it is displayed in an
        ipywidget context. Callbacks will not be triggered on figures
        that are displayed using plot/iplot.

        Parameters
        ----------
        callback
            Callable function that accepts 3 arguments

            - this trace
            - plotly.callbacks.Points object
            - plotly.callbacks.InputDeviceState object

        append : bool
            If False (the default), this callback replaces any previously
            defined on_click callbacks for this trace. If True,
            this callback is appended to the list of any previously defined
            callbacks.

        Returns
        -------
        None

        Examples
        --------

        >>> import plotly.graph_objects as go
        >>> from plotly.callbacks import Points, InputDeviceState
        >>> points, state = Points(), InputDeviceState()

        >>> def click_fn(trace, points, state):
        ...     inds = points.point_inds
        ...     # Do something

        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])
        >>> trace.on_click(click_fn)

        Note: The creation of the `points` and `state` objects is optional,
        it's simply a convenience to help the text editor perform completion
        on the arguments inside `click_fn`
        """
        if not append:
            del self._click_callbacks[:]
        if callback:
            self._click_callbacks.append(callback)

    def _dispatch_on_click(self, points, state):
        """
        Dispatch points and device state all all hover callbacks
        """
        for callback in self._click_callbacks:
            callback(self, points, state)

    def on_selection(self, callback, append=False):
        """
        Register function to be called when the user selects one or more
        points in this trace.

        Note: Callbacks will only be triggered when the trace belongs to a
        instance of plotly.graph_objs.FigureWidget and it is displayed in an
        ipywidget context. Callbacks will not be triggered on figures
        that are displayed using plot/iplot.

        Parameters
        ----------
        callback
            Callable function that accepts 4 arguments

            - this trace
            - plotly.callbacks.Points object
            - plotly.callbacks.BoxSelector or plotly.callbacks.LassoSelector

        append : bool
            If False (the default), this callback replaces any previously
            defined on_selection callbacks for this trace. If True,
            this callback is appended to the list of any previously defined
            callbacks.

        Returns
        -------
        None

        Examples
        --------

        >>> import plotly.graph_objects as go
        >>> from plotly.callbacks import Points
        >>> points = Points()

        >>> def selection_fn(trace, points, selector):
        ...     inds = points.point_inds
        ...     # Do something

        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])
        >>> trace.on_selection(selection_fn)

        Note: The creation of the `points` object is optional,
        it's simply a convenience to help the text editor perform completion
        on the `points` arguments inside `selection_fn`
        """
        if not append:
            del self._select_callbacks[:]
        if callback:
            self._select_callbacks.append(callback)

    def _dispatch_on_selection(self, points, selector):
        """
        Dispatch points and selector info to selection callbacks
        """
        if 'selectedpoints' in self:
            self.selectedpoints = points.point_inds
        for callback in self._select_callbacks:
            callback(self, points, selector)

    def on_deselect(self, callback, append=False):
        """
        Register function to be called when the user deselects points
        in this trace using doubleclick.

        Note: Callbacks will only be triggered when the trace belongs to a
        instance of plotly.graph_objs.FigureWidget and it is displayed in an
        ipywidget context. Callbacks will not be triggered on figures
        that are displayed using plot/iplot.

        Parameters
        ----------
        callback
            Callable function that accepts 3 arguments

            - this trace
            - plotly.callbacks.Points object

        append : bool
            If False (the default), this callback replaces any previously
            defined on_deselect callbacks for this trace. If True,
            this callback is appended to the list of any previously defined
            callbacks.

        Returns
        -------
        None

        Examples
        --------

        >>> import plotly.graph_objects as go
        >>> from plotly.callbacks import Points
        >>> points = Points()

        >>> def deselect_fn(trace, points):
        ...     inds = points.point_inds
        ...     # Do something

        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])
        >>> trace.on_deselect(deselect_fn)

        Note: The creation of the `points` object is optional,
        it's simply a convenience to help the text editor perform completion
        on the `points` arguments inside `selection_fn`
        """
        if not append:
            del self._deselect_callbacks[:]
        if callback:
            self._deselect_callbacks.append(callback)

    def _dispatch_on_deselect(self, points):
        """
        Dispatch points info to deselection callbacks
        """
        if 'selectedpoints' in self:
            self.selectedpoints = None
        for callback in self._deselect_callbacks:
            callback(self, points)