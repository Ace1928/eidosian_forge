from __future__ import annotations
import abc
import logging
import signal
import typing

        Sets the signal handler for signal signum.

        The default implementation of :meth:`set_signal_handler`
        is simply a proxy function that calls :func:`signal.signal()`
        and returns the resulting value.

        signum -- signal number
        handler -- function (taking signum as its single argument),
        or `signal.SIG_IGN`, or `signal.SIG_DFL`
        