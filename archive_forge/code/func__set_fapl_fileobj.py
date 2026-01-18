import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
def _set_fapl_fileobj(plist, **kwargs):
    """Set the Python file object driver in a file access property list"""
    plist.set_fileobj_driver(h5fd.fileobj_driver, kwargs.get('fileobj'))