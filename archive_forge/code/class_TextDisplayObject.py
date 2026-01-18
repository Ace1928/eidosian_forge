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
class TextDisplayObject(DisplayObject):
    """Create a text display object given raw data.

    Parameters
    ----------
    data : str or unicode
        The raw data or a URL or file to load the data from.
    url : unicode
        A URL to download the data from.
    filename : unicode
        Path to a local file to load the data from.
    metadata : dict
        Dict of metadata associated to be the object when displayed
    """

    def _check_data(self):
        if self.data is not None and (not isinstance(self.data, str)):
            raise TypeError('%s expects text, not %r' % (self.__class__.__name__, self.data))