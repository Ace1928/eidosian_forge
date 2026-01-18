from ... import lazy_import
from breezy.bzr.smart import request as _mod_request
import breezy
from ... import debug, errors, hooks, trace
from . import message, protocol
def _run_call_hooks(self):
    if not _SmartClient.hooks['call']:
        return
    params = CallHookParams(self.method, self.args, self.body, self.readv_body, self.client._medium)
    for hook in _SmartClient.hooks['call']:
        hook(params)