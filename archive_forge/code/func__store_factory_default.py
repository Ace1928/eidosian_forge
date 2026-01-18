from __future__ import annotations
import hashlib
import os
import sys
import typing as t
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from hmac import HMAC
from pathlib import Path
from base64 import encodebytes
from jupyter_core.application import JupyterApp, base_flags
from traitlets import Any, Bool, Bytes, Callable, Enum, Instance, Integer, Unicode, default, observe
from traitlets.config import LoggingConfigurable, MultipleInstanceError
from . import NO_CONVERT, __version__, read, reads
@default('store_factory')
def _store_factory_default(self):

    def factory():
        if sqlite3 is None:
            self.log.warning('Missing SQLite3, all notebooks will be untrusted!')
            return MemorySignatureStore()
        return SQLiteSignatureStore(self.db_file)
    return factory