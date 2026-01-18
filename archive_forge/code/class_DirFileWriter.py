import os
import tarfile
import tempfile
import warnings
from io import BytesIO
from shutil import copy2, copytree, rmtree
from .. import osutils
from .. import revision as _mod_revision
from .. import transform
from ..controldir import ControlDir
from ..export import export
from ..upstream_import import (NotArchiveType, ZipFileWrapper,
from . import TestCaseInTempDir, TestCaseWithTransport
from .features import UnicodeFilenameFeature
class DirFileWriter:

    def __init__(self, fileobj, mode):
        fileobj.seek(0)
        existing = fileobj.read()
        fileobj.seek(0)
        path = tempfile.mkdtemp(dir=os.getcwd())
        if existing != b'':
            os.rmdir(path)
            copytree(existing, path)
        fileobj.write(path.encode('utf-8'))
        self.root = path

    def add(self, path):
        target_path = os.path.join(self.root, path)
        parent = osutils.dirname(target_path)
        if not os.path.exists(parent):
            os.makedirs(parent)
        kind = osutils.file_kind(path)
        if kind == 'file':
            copy2(path, target_path)
        if kind == 'directory':
            os.mkdir(target_path)

    def close(self):
        pass