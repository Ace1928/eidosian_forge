from __future__ import annotations
import logging # isort:skip
from collections import defaultdict
from inspect import signature
from typing import (
from ..events import Event, ModelEvent
from ..util.functions import get_param_info
class PropertyCallbackManager:
    """ A mixin class to provide an interface for registering and
    triggering callbacks.

    """
    document: Document | None
    _callbacks: dict[str, list[PropertyCallback]]

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._callbacks = {}

    def on_change(self, attr: str, *callbacks: PropertyCallback) -> None:
        """ Add a callback on this object to trigger when ``attr`` changes.

        Args:
            attr (str) : an attribute name on this object
            callback (callable) : a callback function to register

        Returns:
            None

        """
        if len(callbacks) == 0:
            raise ValueError('on_change takes an attribute name and one or more callbacks, got only one parameter')
        _callbacks = self._callbacks.setdefault(attr, [])
        for callback in callbacks:
            if callback in _callbacks:
                continue
            _check_callback(callback, ('attr', 'old', 'new'))
            _callbacks.append(callback)

    def remove_on_change(self, attr: str, *callbacks: PropertyCallback) -> None:
        """ Remove a callback from this object """
        if len(callbacks) == 0:
            raise ValueError('remove_on_change takes an attribute name and one or more callbacks, got only one parameter')
        _callbacks = self._callbacks.setdefault(attr, [])
        for callback in callbacks:
            _callbacks.remove(callback)

    def trigger(self, attr: str, old: Any, new: Any, hint: DocumentPatchedEvent | None=None, setter: Setter | None=None) -> None:
        """ Trigger callbacks for ``attr`` on this object.

        Args:
            attr (str) :
            old (object) :
            new (object) :

        Returns:
            None

        """

        def invoke() -> None:
            callbacks = self._callbacks.get(attr)
            if callbacks:
                for callback in callbacks:
                    callback(attr, old, new)
        if self.document is not None:
            from ..model import Model
            self.document.callbacks.notify_change(cast(Model, self), attr, old, new, hint, setter, invoke)
        else:
            invoke()