imported from breezy.bzr.smart.
from io import BytesIO
from .. import config, debug, errors, trace, transport, urlutils
from ..bzr import remote
from ..bzr.smart import client, medium
def _call2(self, method, *args):
    """Call a method on the remote server."""
    try:
        return self._client.call(method, *args)
    except errors.ErrorFromSmartServer as err:
        if args:
            context = {'relpath': args[0].decode('utf-8')}
        else:
            context = {}
        self._translate_error(err, **context)