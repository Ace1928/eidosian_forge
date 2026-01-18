import copy
import logging
from collections import deque, namedtuple
from botocore.compat import accepts_kwargs
from botocore.utils import EVENT_ALIASES
class BaseEventHooks:

    def emit(self, event_name, **kwargs):
        """Call all handlers subscribed to an event.

        :type event_name: str
        :param event_name: The name of the event to emit.

        :type **kwargs: dict
        :param **kwargs: Arbitrary kwargs to pass through to the
            subscribed handlers.  The ``event_name`` will be injected
            into the kwargs so it's not necessary to add this to **kwargs.

        :rtype: list of tuples
        :return: A list of ``(handler_func, handler_func_return_value)``

        """
        return []

    def register(self, event_name, handler, unique_id=None, unique_id_uses_count=False):
        """Register an event handler for a given event.

        If a ``unique_id`` is given, the handler will not be registered
        if a handler with the ``unique_id`` has already been registered.

        Handlers are called in the order they have been registered.
        Note handlers can also be registered with ``register_first()``
        and ``register_last()``.  All handlers registered with
        ``register_first()`` are called before handlers registered
        with ``register()`` which are called before handlers registered
        with ``register_last()``.

        """
        self._verify_and_register(event_name, handler, unique_id, register_method=self._register, unique_id_uses_count=unique_id_uses_count)

    def register_first(self, event_name, handler, unique_id=None, unique_id_uses_count=False):
        """Register an event handler to be called first for an event.

        All event handlers registered with ``register_first()`` will
        be called before handlers registered with ``register()`` and
        ``register_last()``.

        """
        self._verify_and_register(event_name, handler, unique_id, register_method=self._register_first, unique_id_uses_count=unique_id_uses_count)

    def register_last(self, event_name, handler, unique_id=None, unique_id_uses_count=False):
        """Register an event handler to be called last for an event.

        All event handlers registered with ``register_last()`` will be called
        after handlers registered with ``register_first()`` and ``register()``.

        """
        self._verify_and_register(event_name, handler, unique_id, register_method=self._register_last, unique_id_uses_count=unique_id_uses_count)

    def _verify_and_register(self, event_name, handler, unique_id, register_method, unique_id_uses_count):
        self._verify_is_callable(handler)
        self._verify_accept_kwargs(handler)
        register_method(event_name, handler, unique_id, unique_id_uses_count)

    def unregister(self, event_name, handler=None, unique_id=None, unique_id_uses_count=False):
        """Unregister an event handler for a given event.

        If no ``unique_id`` was given during registration, then the
        first instance of the event handler is removed (if the event
        handler has been registered multiple times).

        """
        pass

    def _verify_is_callable(self, func):
        if not callable(func):
            raise ValueError('Event handler %s must be callable.' % func)

    def _verify_accept_kwargs(self, func):
        """Verifies a callable accepts kwargs

        :type func: callable
        :param func: A callable object.

        :returns: True, if ``func`` accepts kwargs, otherwise False.

        """
        try:
            if not accepts_kwargs(func):
                raise ValueError(f'Event handler {func} must accept keyword arguments (**kwargs)')
        except TypeError:
            return False