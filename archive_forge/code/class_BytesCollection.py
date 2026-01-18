from contextlib import ExitStack
import logging
import os
import warnings
from collections import OrderedDict
from fiona import compat, vfs
from fiona.ogrext import Iterator, ItemsIterator, KeysIterator
from fiona.ogrext import Session, WritingSession
from fiona.ogrext import buffer_to_virtual_file, remove_virtual_file, GEOMETRY_TYPES
from fiona.errors import (
from fiona.logutils import FieldSkipLogFilter
from fiona.crs import CRS
from fiona._env import get_gdal_release_name, get_gdal_version_tuple
from fiona.env import env_ctx_if_needed
from fiona.errors import FionaDeprecationWarning
from fiona.drvsupport import (
from fiona.path import Path, vsi_path, parse_path
class BytesCollection(Collection):
    """BytesCollection takes a buffer of bytes and maps that to
    a virtual file that can then be opened by fiona.
    """

    def __init__(self, bytesbuf, **kwds):
        """Takes buffer of bytes whose contents is something we'd like
        to open with Fiona and maps it to a virtual file.

        """
        self._closed = True
        if not isinstance(bytesbuf, bytes):
            raise ValueError('input buffer must be bytes')
        self.bytesbuf = bytesbuf
        filetype = get_filetype(self.bytesbuf)
        ext = ''
        if filetype == 'zip':
            ext = '.zip'
        elif kwds.get('driver') == 'GeoJSON':
            ext = '.json'
        self.virtual_file = buffer_to_virtual_file(self.bytesbuf, ext=ext)
        super().__init__(self.virtual_file, vsi=filetype, **kwds)
        self._closed = False

    def close(self):
        """Removes the virtual file associated with the class."""
        super().close()
        if self.virtual_file:
            remove_virtual_file(self.virtual_file)
            self.virtual_file = None
            self.bytesbuf = None

    def __repr__(self):
        return "<{} BytesCollection '{}', mode '{}' at {}>".format(self.closed and 'closed' or 'open', self.path + ':' + str(self.name), self.mode, hex(id(self)))