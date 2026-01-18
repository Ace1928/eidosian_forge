import contextvars
import os
import socket
import subprocess
import sys
import threading
from . import format_helpers
class AbstractServer:
    """Abstract server returned by create_server()."""

    def close(self):
        """Stop serving.  This leaves existing connections open."""
        raise NotImplementedError

    def get_loop(self):
        """Get the event loop the Server object is attached to."""
        raise NotImplementedError

    def is_serving(self):
        """Return True if the server is accepting connections."""
        raise NotImplementedError

    async def start_serving(self):
        """Start accepting connections.

        This method is idempotent, so it can be called when
        the server is already being serving.
        """
        raise NotImplementedError

    async def serve_forever(self):
        """Start accepting connections until the coroutine is cancelled.

        The server is closed when the coroutine is cancelled.
        """
        raise NotImplementedError

    async def wait_closed(self):
        """Coroutine to wait until service is closed."""
        raise NotImplementedError

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.close()
        await self.wait_closed()