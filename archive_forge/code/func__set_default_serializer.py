from __future__ import annotations
import codecs
import os
import pickle
import sys
from collections import namedtuple
from contextlib import contextmanager
from io import BytesIO
from .exceptions import (ContentDisallowed, DecodeError, EncodeError,
from .utils.compat import entrypoints
from .utils.encoding import bytes_to_str, str_to_bytes
def _set_default_serializer(self, name):
    """Set the default serialization method used by this library.

        Arguments:
        ---------
            name (str): The name of the registered serialization method.
                For example, `json` (default), `pickle`, `yaml`, `msgpack`,
                or any custom methods registered using :meth:`register`.

        Raises
        ------
            SerializerNotInstalled: If the serialization method
                requested is not available.
        """
    try:
        self._default_content_type, self._default_content_encoding, self._default_encode = self._encoders[name]
    except KeyError:
        raise SerializerNotInstalled(f'No encoder installed for {name}')