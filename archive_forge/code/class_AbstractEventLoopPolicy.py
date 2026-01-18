import contextvars
import os
import socket
import subprocess
import sys
import threading
from . import format_helpers
class AbstractEventLoopPolicy:
    """Abstract policy for accessing the event loop."""

    def get_event_loop(self):
        """Get the event loop for the current context.

        Returns an event loop object implementing the AbstractEventLoop interface,
        or raises an exception in case no event loop has been set for the
        current context and the current policy does not specify to create one.

        It should never return None."""
        raise NotImplementedError

    def set_event_loop(self, loop):
        """Set the event loop for the current context to loop."""
        raise NotImplementedError

    def new_event_loop(self):
        """Create and return a new event loop object according to this
        policy's rules. If there's need to set this loop as the event loop for
        the current context, set_event_loop must be called explicitly."""
        raise NotImplementedError

    def get_child_watcher(self):
        """Get the watcher for child processes."""
        raise NotImplementedError

    def set_child_watcher(self, watcher):
        """Set the watcher for child processes."""
        raise NotImplementedError