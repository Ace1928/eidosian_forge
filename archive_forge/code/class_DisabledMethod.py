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
class DisabledMethod(errors.InternalBzrError):
    _fmt = "The smart server method '%(class_name)s' is disabled."

    def __init__(self, class_name):
        errors.BzrError.__init__(self)
        self.class_name = class_name