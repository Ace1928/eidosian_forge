import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def _accessible_normalized_filename(path):
    """Get the unicode normalized path, and if you can access the file.

    On platforms where the system normalizes filenames (Mac OSX),
    you can access a file by any path which will normalize correctly.
    On platforms where the system does not normalize filenames
    (everything else), you have to access a file by its exact path.

    Internally, bzr only supports NFC normalization, since that is
    the standard for XML documents.

    So return the normalized path, and a flag indicating if the file
    can be accessed by that path.
    """
    if isinstance(path, bytes):
        path = path.decode(sys.getfilesystemencoding())
    return (unicodedata.normalize('NFC', path), True)