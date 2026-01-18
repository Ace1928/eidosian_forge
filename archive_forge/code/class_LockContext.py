from ncclient.xml_ import *
from ncclient.operations.rpc import RaiseMode, RPC
class LockContext(object):
    """A context manager for the :class:`Lock` / :class:`Unlock` pair of RPC's.

    Any `rpc-error` will be raised as an exception.

    Initialise with (:class:`Session <ncclient.transport.Session>`) instance and lock target.
    """

    def __init__(self, session, device_handler, target):
        self.session = session
        self.target = target
        self.device_handler = device_handler

    def __enter__(self):
        Lock(self.session, self.device_handler, raise_mode=RaiseMode.ERRORS).request(self.target)
        return self

    def __exit__(self, *args):
        Unlock(self.session, self.device_handler, raise_mode=RaiseMode.ERRORS).request(self.target)
        return False