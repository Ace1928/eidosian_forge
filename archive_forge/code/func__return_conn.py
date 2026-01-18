from __future__ import annotations
from collections import deque
import dataclasses
from enum import Enum
import threading
import time
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
def _return_conn(self, record: ConnectionPoolEntry) -> None:
    """Given a _ConnectionRecord, return it to the :class:`_pool.Pool`.

        This method is called when an instrumented DBAPI connection
        has its ``close()`` method called.

        """
    self._do_return_conn(record)