import logging
import sys
from contextlib import contextmanager
from IPython.core.interactiveshell import InteractiveShellABC
from traitlets import Any, Enum, Instance, List, Type, default
from ipykernel.ipkernel import IPythonKernel
from ipykernel.jsonutil import json_clean
from ipykernel.zmqshell import ZMQInteractiveShell
from ..iostream import BackgroundSocket, IOPubThread, OutStream
from .constants import INPROCESS_KEY
from .socket import DummySocket
def _input_request(self, prompt, ident, parent, password=False):
    self.raw_input_str = None
    sys.stderr.flush()
    sys.stdout.flush()
    content = json_clean(dict(prompt=prompt, password=password))
    assert self.session is not None
    msg = self.session.msg('input_request', content, parent)
    for frontend in self.frontends:
        assert frontend is not None
        if frontend.session.session == parent['header']['session']:
            frontend.stdin_channel.call_handlers(msg)
            break
    else:
        logging.error('No frontend found for raw_input request')
        return ''
    while self.raw_input_str is None:
        frontend.stdin_channel.process_events()
    return self.raw_input_str