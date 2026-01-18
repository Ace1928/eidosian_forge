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
class SignatureStore:
    """Base class for a signature store."""

    def store_signature(self, digest, algorithm):
        """Implement in subclass to store a signature.

        Should not raise if the signature is already stored.
        """
        raise NotImplementedError

    def check_signature(self, digest, algorithm):
        """Implement in subclass to check if a signature is known.

        Return True for a known signature, False for unknown.
        """
        raise NotImplementedError

    def remove_signature(self, digest, algorithm):
        """Implement in subclass to delete a signature.

        Should not raise if the signature is not stored.
        """
        raise NotImplementedError

    def close(self):
        """Close any open connections this store may use.

        If the store maintains any open connections (e.g. to a database),
        they should be closed.
        """