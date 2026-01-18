import threading
from _thread import get_ident
from ... import branch as _mod_branch
from ... import debug, errors, osutils, registry, revision, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...lazy_import import lazy_import
from breezy.bzr import bzrdir
from breezy.bzr.bundle import serializer
import tempfile
def accept_body(self, bytes):
    """Accept body data."""
    if self._command is None:
        return
    self._run_handler_code(self._command.do_chunk, (bytes,), {})
    if 'hpss' in debug.debug_flags:
        self._trace('accept body', '%d bytes' % (len(bytes),), bytes)