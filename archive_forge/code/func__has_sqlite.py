from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
def _has_sqlite(self):
    from sqlalchemy import create_engine
    try:
        create_engine('sqlite://')
        return True
    except ImportError:
        return False