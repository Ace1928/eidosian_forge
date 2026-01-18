from __future__ import annotations
import io
from functools import partial
from typing import (
import trio
from ._util import async_wraps
from .abc import AsyncResource
Like :meth:`io.IOBase.close`, but async.

        This is also shielded from cancellation; if a cancellation scope is
        cancelled, the wrapped file object will still be safely closed.

        