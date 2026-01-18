from ._api import request, stream
from ._async import (
from ._backends.base import (
from ._backends.mock import AsyncMockBackend, AsyncMockStream, MockBackend, MockStream
from ._backends.sync import SyncBackend
from ._exceptions import (
from ._models import URL, Origin, Request, Response
from ._ssl import default_ssl_context
from ._sync import (
class TrioBackend:

    def __init__(self, *args, **kwargs):
        msg = "Attempted to use 'httpcore.TrioBackend' but 'trio' is not installed."
        raise RuntimeError(msg)