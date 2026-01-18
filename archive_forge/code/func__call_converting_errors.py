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
def _call_converting_errors(self, callable, args, kwargs):
    """Call callable converting errors to Response objects."""
    try:
        self._command.setup_jail()
        try:
            return callable(*args, **kwargs)
        finally:
            self._command.teardown_jail()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as err:
        err_struct = _translate_error(err)
        return FailedSmartServerResponse(err_struct)