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
def _run_handler_code(self, callable, args, kwargs):
    """Run some handler specific code 'callable'.

        If a result is returned, it is considered to be the commands response,
        and finished_reading is set true, and its assigned to self.response.

        Any exceptions caught are translated and a response object created
        from them.
        """
    result = self._call_converting_errors(callable, args, kwargs)
    if result is not None:
        self.response = result
        self.finished_reading = True