import errno
import os
import re
import stat
import tarfile
import zipfile
from io import BytesIO
from . import urlutils
from .bzr import generate_ids
from .controldir import ControlDir, is_control_filename
from .errors import BzrError, CommandError, NotBranchError
from .osutils import (basename, file_iterator, file_kind, isdir, pathjoin,
from .trace import warning
from .transform import resolve_conflicts
from .transport import NoSuchFile, get_transport
from .workingtree import WorkingTree
class ZipInfoWrapper:

    def __init__(self, zipfile, info):
        self.info = info
        self.type = None
        self.name = info.filename
        self.zipfile = zipfile
        self.mode = 438

    def isdir(self):
        return bool(self.name.endswith('/'))

    def isreg(self):
        return not self.isdir()