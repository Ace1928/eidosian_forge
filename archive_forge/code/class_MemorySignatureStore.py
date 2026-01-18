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
class MemorySignatureStore(SignatureStore):
    """Non-persistent storage of signatures in memory."""
    cache_size = 65535

    def __init__(self):
        """Initialize a memory signature store."""
        self.data = OrderedDict()

    def store_signature(self, digest, algorithm):
        """Store a signature."""
        key = (digest, algorithm)
        self.data.pop(key, None)
        self.data[key] = None
        self._maybe_cull()

    def _maybe_cull(self):
        """If more than cache_size signatures are stored, delete the oldest 25%"""
        if len(self.data) < self.cache_size:
            return
        for _ in range(len(self.data) // 4):
            self.data.popitem(last=False)

    def check_signature(self, digest, algorithm):
        """Check a signature."""
        key = (digest, algorithm)
        if key in self.data:
            del self.data[key]
            self.data[key] = None
            return True
        return False

    def remove_signature(self, digest, algorithm):
        """Remove a signature."""
        self.data.pop((digest, algorithm), None)