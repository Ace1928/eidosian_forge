from __future__ import annotations
import errno
import logging
import os
from typing import Any, Awaitable, Callable, NoReturn, TypeVar
import trio
Listen for incoming connections on ``listeners``, and for each one
    start a task running ``handler(stream)``.

    .. warning::

       If ``handler`` raises an exception, then this function doesn't do
       anything special to catch it – so by default the exception will
       propagate out and crash your server. If you don't want this, then catch
       exceptions inside your ``handler``, or use a ``handler_nursery`` object
       that responds to exceptions in some other way.

    Args:

      handler: An async callable, that will be invoked like
          ``handler_nursery.start_soon(handler, stream)`` for each incoming
          connection.

      listeners: A list of :class:`~trio.abc.Listener` objects.
          :func:`serve_listeners` takes responsibility for closing them.

      handler_nursery: The nursery used to start handlers, or any object with
          a ``start_soon`` method. If ``None`` (the default), then
          :func:`serve_listeners` will create a new nursery internally and use
          that.

      task_status: This function can be used with ``nursery.start``, which
          will return ``listeners``.

    Returns:

      This function never returns unless cancelled.

    Resource handling:

      If ``handler`` neglects to close the ``stream``, then it will be closed
      using :func:`trio.aclose_forcefully`.

    Error handling:

      Most errors coming from :meth:`~trio.abc.Listener.accept` are allowed to
      propagate out (crashing the server in the process). However, some errors –
      those which indicate that the server is temporarily overloaded – are
      handled specially. These are :class:`OSError`\s with one of the following
      errnos:

      * ``EMFILE``: process is out of file descriptors
      * ``ENFILE``: system is out of file descriptors
      * ``ENOBUFS``, ``ENOMEM``: the kernel hit some sort of memory limitation
        when trying to create a socket object

      When :func:`serve_listeners` gets one of these errors, then it:

      * Logs the error to the standard library logger ``trio.serve_listeners``
        (level = ERROR, with exception information included). By default this
        causes it to be printed to stderr.
      * Waits 100 ms before calling ``accept`` again, in hopes that the
        system will recover.

    