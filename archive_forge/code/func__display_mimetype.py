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
def _display_mimetype(mimetype, objs, raw=False, metadata=None):
    """internal implementation of all display_foo methods

    Parameters
    ----------
    mimetype : str
        The mimetype to be published (e.g. 'image/png')
    *objs : object
        The Python objects to display, or if raw=True raw text data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    if metadata:
        metadata = {mimetype: metadata}
    if raw:
        objs = [{mimetype: obj} for obj in objs]
    display_functions.display(*objs, raw=raw, metadata=metadata, include=[mimetype])