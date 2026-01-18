from __future__ import annotations
import logging # isort:skip
import sys
import threading
from collections import defaultdict
from traceback import format_exception
from typing import (
import tornado
from tornado import gen
from ..core.types import ID
class _CallbackGroup:
    """ A collection of callbacks added to a Tornado IOLoop that can be removed
    as a group.

    """
    _next_tick_callback_removers: Removers
    _timeout_callback_removers: Removers
    _periodic_callback_removers: Removers
    _next_tick_removers_by_callable: RemoversByCallable
    _timeout_removers_by_callable: RemoversByCallable
    _periodic_removers_by_callable: RemoversByCallable
    _loop: IOLoop

    def __init__(self, io_loop: IOLoop) -> None:
        self._loop = io_loop
        self._next_tick_callback_removers = {}
        self._timeout_callback_removers = {}
        self._periodic_callback_removers = {}
        self._removers_lock = threading.Lock()
        self._next_tick_removers_by_callable = defaultdict(set)
        self._timeout_removers_by_callable = defaultdict(set)
        self._periodic_removers_by_callable = defaultdict(set)

    def remove_all_callbacks(self) -> None:
        """ Removes all registered callbacks.

        """
        for cb_id in list(self._next_tick_callback_removers):
            self.remove_next_tick_callback(cb_id)
        for cb_id in list(self._timeout_callback_removers):
            self.remove_timeout_callback(cb_id)
        for cb_id in list(self._periodic_callback_removers):
            self.remove_periodic_callback(cb_id)

    def _get_removers_ids_by_callable(self, removers: Removers) -> RemoversByCallable:
        if removers is self._next_tick_callback_removers:
            return self._next_tick_removers_by_callable
        elif removers is self._timeout_callback_removers:
            return self._timeout_removers_by_callable
        elif removers is self._periodic_callback_removers:
            return self._periodic_removers_by_callable
        else:
            raise RuntimeError('Unhandled removers', removers)

    def _assign_remover(self, callback: Callback, callback_id: ID, removers: Removers, remover: Remover) -> None:
        with self._removers_lock:
            if callback_id in removers:
                raise ValueError('A callback of the same type has already been added with this ID')
            removers[callback_id] = remover

    def _execute_remover(self, callback_id: ID, removers: Removers) -> None:
        try:
            with self._removers_lock:
                remover = removers.pop(callback_id)
                for cb, cb_ids in list(self._get_removers_ids_by_callable(removers).items()):
                    try:
                        cb_ids.remove(callback_id)
                        if not cb_ids:
                            del self._get_removers_ids_by_callable(removers)[cb]
                    except KeyError:
                        pass
        except KeyError:
            raise ValueError("Removing a callback twice (or after it's already been run)")
        remover()

    def add_next_tick_callback(self, callback: Callback, callback_id: ID) -> ID:
        """ Adds a callback to be run on the nex

        The passed-in ID can be used with remove_next_tick_callback.

        """

        def wrapper() -> None | Awaitable[None]:
            if wrapper.removed:
                return None
            self.remove_next_tick_callback(callback_id)
            return callback()
        wrapper.removed = False

        def remover() -> None:
            wrapper.removed = True
        self._assign_remover(callback, callback_id, self._next_tick_callback_removers, remover)
        self._loop.add_callback(wrapper)
        return callback_id

    def remove_next_tick_callback(self, callback_id: ID) -> None:
        """ Removes a callback added with add_next_tick_callback.

        """
        self._execute_remover(callback_id, self._next_tick_callback_removers)

    def add_timeout_callback(self, callback: CallbackSync, timeout_milliseconds: int, callback_id: ID) -> ID:
        """ Adds a callback to be run once after timeout_milliseconds.

        The passed-in ID can be used with remove_timeout_callback.

        """

        def wrapper() -> None:
            self.remove_timeout_callback(callback_id)
            return callback()
        handle: object | None = None

        def remover() -> None:
            if handle is not None:
                self._loop.remove_timeout(handle)
        self._assign_remover(callback, callback_id, self._timeout_callback_removers, remover)
        handle = self._loop.call_later(timeout_milliseconds / 1000.0, wrapper)
        return callback_id

    def remove_timeout_callback(self, callback_id: ID) -> None:
        """ Removes a callback added with add_timeout_callback, before it runs.

        """
        self._execute_remover(callback_id, self._timeout_callback_removers)

    def add_periodic_callback(self, callback: Callback, period_milliseconds: int, callback_id: ID) -> None:
        """ Adds a callback to be run every period_milliseconds until it is removed.

        The passed-in ID can be used with remove_periodic_callback.

        """
        cb = _AsyncPeriodic(callback, period_milliseconds, io_loop=self._loop)
        self._assign_remover(callback, callback_id, self._periodic_callback_removers, cb.stop)
        cb.start()

    def remove_periodic_callback(self, callback_id: ID) -> None:
        """ Removes a callback added with add_periodic_callback.

        """
        self._execute_remover(callback_id, self._periodic_callback_removers)