import errno
import os.path
import socket
import sys
import threading
import time
from ... import errors, trace
from ... import transport as _mod_transport
from ...hooks import Hooks
from ...i18n import gettext
from ...lazy_import import lazy_import
from breezy.bzr.smart import (
from breezy.transport import (
from breezy import (
def _make_smart_server(self, host, port, inet, timeout):
    if timeout is None:
        c = config.GlobalStack()
        timeout = c.get('serve.client_timeout')
    if inet:
        stdin, stdout = self._get_stdin_stdout()
        smart_server = medium.SmartServerPipeStreamMedium(stdin, stdout, self.transport, timeout=timeout)
    else:
        if host is None:
            host = medium.BZR_DEFAULT_INTERFACE
        if port is None:
            port = medium.BZR_DEFAULT_PORT
        smart_server = SmartTCPServer(self.transport, client_timeout=timeout)
        smart_server.start_server(host, port)
        trace.note(gettext('listening on port: %s'), str(smart_server.port))
    self.smart_server = smart_server