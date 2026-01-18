from binascii import b2a_base64, hexlify
import html
import json
import mimetypes
import os
import struct
import warnings
from copy import deepcopy
from os.path import splitext
from pathlib import Path, PurePath
from IPython.utils.py3compat import cast_unicode
from IPython.testing.skipdoctest import skip_doctest
from . import display_functions
from warnings import warn
class DisplayObject(object):
    """An object that wraps data to be displayed."""
    _read_flags = 'r'
    _show_mem_addr = False
    metadata = None

    def __init__(self, data=None, url=None, filename=None, metadata=None):
        """Create a display object given raw data.

        When this object is returned by an expression or passed to the
        display function, it will result in the data being displayed
        in the frontend. The MIME type of the data should match the
        subclasses used, so the Png subclass should be used for 'image/png'
        data. If the data is a URL, the data will first be downloaded
        and then displayed.

        Parameters
        ----------
        data : unicode, str or bytes
            The raw data or a URL or file to load the data from
        url : unicode
            A URL to download the data from.
        filename : unicode
            Path to a local file to load the data from.
        metadata : dict
            Dict of metadata associated to be the object when displayed
        """
        if isinstance(data, (Path, PurePath)):
            data = str(data)
        if data is not None and isinstance(data, str):
            if data.startswith('http') and url is None:
                url = data
                filename = None
                data = None
            elif _safe_exists(data) and filename is None:
                url = None
                filename = data
                data = None
        self.url = url
        self.filename = filename
        self.data = data
        if metadata is not None:
            self.metadata = metadata
        elif self.metadata is None:
            self.metadata = {}
        self.reload()
        self._check_data()

    def __repr__(self):
        if not self._show_mem_addr:
            cls = self.__class__
            r = '<%s.%s object>' % (cls.__module__, cls.__name__)
        else:
            r = super(DisplayObject, self).__repr__()
        return r

    def _check_data(self):
        """Override in subclasses if there's something to check."""
        pass

    def _data_and_metadata(self):
        """shortcut for returning metadata with shape information, if defined"""
        if self.metadata:
            return (self.data, deepcopy(self.metadata))
        else:
            return self.data

    def reload(self):
        """Reload the raw data from file or URL."""
        if self.filename is not None:
            encoding = None if 'b' in self._read_flags else 'utf-8'
            with open(self.filename, self._read_flags, encoding=encoding) as f:
                self.data = f.read()
        elif self.url is not None:
            from urllib.request import urlopen
            response = urlopen(self.url)
            data = response.read()
            encoding = None
            if 'content-type' in response.headers:
                for sub in response.headers['content-type'].split(';'):
                    sub = sub.strip()
                    if sub.startswith('charset'):
                        encoding = sub.split('=')[-1].strip()
                        break
            if 'content-encoding' in response.headers:
                if 'gzip' in response.headers['content-encoding']:
                    import gzip
                    from io import BytesIO
                    with gzip.open(BytesIO(data), 'rt', encoding=encoding or 'utf-8') as fp:
                        encoding = None
                        data = fp.read()
            if encoding:
                self.data = data.decode(encoding, 'replace')
            else:
                self.data = data