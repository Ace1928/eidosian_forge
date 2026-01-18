from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
class CallHookParams:

    def __init__(self, method, args, body, readv_body, medium):
        self.method = method
        self.args = args
        self.body = body
        self.readv_body = readv_body
        self.medium = medium

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if v is not None}
        return '<{} {!r}>'.format(self.__class__.__name__, attrs)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other