import logging
from vine import ensure_promise, promise
from .exceptions import AMQPNotImplementedError, RecoverableConnectionError
from .serialization import dumps, loads
Close this Channel or Connection.