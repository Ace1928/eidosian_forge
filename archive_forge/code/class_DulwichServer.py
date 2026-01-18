import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
class DulwichServer:
    """Start the TCPGitServer with Swift backend."""

    def __init__(self, backend, port) -> None:
        self.port = port
        self.backend = backend

    def run(self):
        self.server = server.TCPGitServer(self.backend, 'localhost', port=self.port)
        self.job = gevent.spawn(self.server.serve_forever)

    def stop(self):
        self.server.shutdown()
        gevent.joinall((self.job,))