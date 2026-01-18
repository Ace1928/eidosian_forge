import copy
import re
import threading
import time
import warnings
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type, Union
from redis._parsers.encoders import Encoder
from redis._parsers.helpers import (
from redis.commands import (
from redis.connection import (
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
def _disconnect_reset_raise(self, conn, error) -> None:
    """
        Close the connection, reset watching state and
        raise an exception if we were watching,
        if retry_on_error is not set or the error is not one
        of the specified error types.
        """
    conn.disconnect()
    if self.watching:
        self.reset()
        raise WatchError('A ConnectionError occurred on while watching one or more keys')
    if conn.retry_on_error is None or isinstance(error, tuple(conn.retry_on_error)) is False:
        self.reset()
        raise