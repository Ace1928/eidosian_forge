from __future__ import annotations
import builtins
import sys
import typing as t
from IPython.core.displayhook import DisplayHook
from jupyter_client.session import Session, extract_header
from traitlets import Any, Dict, Instance
from ipykernel.jsonutil import encode_images, json_clean
class ZMQDisplayHook:
    """A simple displayhook that publishes the object's repr over a ZeroMQ
    socket."""
    topic = b'execute_result'

    def __init__(self, session, pub_socket):
        """Initialize the hook."""
        self.session = session
        self.pub_socket = pub_socket
        self.parent_header = {}

    def get_execution_count(self):
        """This method is replaced in kernelapp"""
        return 0

    def __call__(self, obj):
        """Handle a hook call."""
        if obj is None:
            return
        builtins._ = obj
        sys.stdout.flush()
        sys.stderr.flush()
        contents = {'execution_count': self.get_execution_count(), 'data': {'text/plain': repr(obj)}, 'metadata': {}}
        self.session.send(self.pub_socket, 'execute_result', contents, parent=self.parent_header, ident=self.topic)

    def set_parent(self, parent):
        """Set the parent header."""
        self.parent_header = extract_header(parent)