from __future__ import annotations
from functools import wraps
import zmq
class _SocketDecorator(_Decorator):
    """Decorator subclass for sockets

    Gets the context from other args.
    """

    def process_decorator_args(self, *args, **kwargs):
        """Also grab context_name out of kwargs"""
        kw_name, args, kwargs = super().process_decorator_args(*args, **kwargs)
        self.context_name = kwargs.pop('context_name', 'context')
        return (kw_name, args, kwargs)

    def get_target(self, *args, **kwargs):
        """Get context, based on call-time args"""
        context = self._get_context(*args, **kwargs)
        return context.socket

    def _get_context(self, *args, **kwargs):
        """
        Find the ``zmq.Context`` from ``args`` and ``kwargs`` at call time.

        First, if there is an keyword argument named ``context`` and it is a
        ``zmq.Context`` instance , we will take it.

        Second, we check all the ``args``, take the first ``zmq.Context``
        instance.

        Finally, we will provide default Context -- ``zmq.Context.instance``

        :return: a ``zmq.Context`` instance
        """
        if self.context_name in kwargs:
            ctx = kwargs[self.context_name]
            if isinstance(ctx, zmq.Context):
                return ctx
        for arg in args:
            if isinstance(arg, zmq.Context):
                return arg
        return zmq.Context.instance()