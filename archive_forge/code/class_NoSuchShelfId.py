import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
class NoSuchShelfId(errors.BzrError):
    _fmt = 'No changes are shelved with id "%(shelf_id)d".'

    def __init__(self, shelf_id):
        errors.BzrError.__init__(self, shelf_id=shelf_id)