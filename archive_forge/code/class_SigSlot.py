import ast
import contextlib
import logging
import os
import re
from typing import ClassVar, Sequence
import panel as pn
from .core import OpenFile, get_filesystem_class, split_protocol
from .registry import known_implementations
class SigSlot:
    """Signal-slot mixin, for Panel event passing

    Include this class in a widget manager's superclasses to be able to
    register events and callbacks on Panel widgets managed by that class.

    The method ``_register`` should be called as widgets are added, and external
    code should call ``connect`` to associate callbacks.

    By default, all signals emit a DEBUG logging statement.
    """
    signals: ClassVar[Sequence[str]] = []
    slots: ClassVar[Sequence[str]] = []

    def __init__(self):
        self._ignoring_events = False
        self._sigs = {}
        self._map = {}
        self._setup()

    def _setup(self):
        """Create GUI elements and register signals"""
        self.panel = pn.pane.PaneBase()

    def _register(self, widget, name, thing='value', log_level=logging.DEBUG, auto=False):
        """Watch the given attribute of a widget and assign it a named event

        This is normally called at the time a widget is instantiated, in the
        class which owns it.

        Parameters
        ----------
        widget : pn.layout.Panel or None
            Widget to watch. If None, an anonymous signal not associated with
            any widget.
        name : str
            Name of this event
        thing : str
            Attribute of the given widget to watch
        log_level : int
            When the signal is triggered, a logging event of the given level
            will be fired in the dfviz logger.
        auto : bool
            If True, automatically connects with a method in this class of the
            same name.
        """
        if name not in self.signals:
            raise ValueError(f'Attempt to assign an undeclared signal: {name}')
        self._sigs[name] = {'widget': widget, 'callbacks': [], 'thing': thing, 'log': log_level}
        wn = '-'.join([getattr(widget, 'name', str(widget)) if widget is not None else 'none', thing])
        self._map[wn] = name
        if widget is not None:
            widget.param.watch(self._signal, thing, onlychanged=True)
        if auto and hasattr(self, name):
            self.connect(name, getattr(self, name))

    def _repr_mimebundle_(self, *args, **kwargs):
        """Display in a notebook or a server"""
        try:
            return self.panel._repr_mimebundle_(*args, **kwargs)
        except (ValueError, AttributeError):
            raise NotImplementedError('Panel does not seem to be set up properly')

    def connect(self, signal, slot):
        """Associate call back with given event

        The callback must be a function which takes the "new" value of the
        watched attribute as the only parameter. If the callback return False,
        this cancels any further processing of the given event.

        Alternatively, the callback can be a string, in which case it means
        emitting the correspondingly-named event (i.e., connect to self)
        """
        self._sigs[signal]['callbacks'].append(slot)

    def _signal(self, event):
        """This is called by a an action on a widget

        Within an self.ignore_events context, nothing happens.

        Tests can execute this method by directly changing the values of
        widget components.
        """
        if not self._ignoring_events:
            wn = '-'.join([event.obj.name, event.name])
            if wn in self._map and self._map[wn] in self._sigs:
                self._emit(self._map[wn], event.new)

    @contextlib.contextmanager
    def ignore_events(self):
        """Temporarily turn off events processing in this instance

        (does not propagate to children)
        """
        self._ignoring_events = True
        try:
            yield
        finally:
            self._ignoring_events = False

    def _emit(self, sig, value=None):
        """An event happened, call its callbacks

        This method can be used in tests to simulate message passing without
        directly changing visual elements.

        Calling of callbacks will halt whenever one returns False.
        """
        logger.log(self._sigs[sig]['log'], f'{sig}: {value}')
        for callback in self._sigs[sig]['callbacks']:
            if isinstance(callback, str):
                self._emit(callback)
            else:
                try:
                    ret = callback(value)
                    if ret is False:
                        break
                except Exception as e:
                    logger.exception('Exception (%s) while executing callback for signal: %s', e, sig)

    def show(self, threads=False):
        """Open a new browser tab and display this instance's interface"""
        self.panel.show(threads=threads, verbose=False)
        return self