import os
import shutil
import stat
import struct
import sys
import tempfile
from io import BytesIO
from dulwich.tests import TestCase, skipIf
from ..index import (
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..repo import Repo
class TestValidatePathElement(TestCase):

    def test_default(self):
        self.assertTrue(validate_path_element_default(b'bla'))
        self.assertTrue(validate_path_element_default(b'.bla'))
        self.assertFalse(validate_path_element_default(b'.git'))
        self.assertFalse(validate_path_element_default(b'.giT'))
        self.assertFalse(validate_path_element_default(b'..'))
        self.assertTrue(validate_path_element_default(b'git~1'))

    def test_ntfs(self):
        self.assertTrue(validate_path_element_ntfs(b'bla'))
        self.assertTrue(validate_path_element_ntfs(b'.bla'))
        self.assertFalse(validate_path_element_ntfs(b'.git'))
        self.assertFalse(validate_path_element_ntfs(b'.giT'))
        self.assertFalse(validate_path_element_ntfs(b'..'))
        self.assertFalse(validate_path_element_ntfs(b'git~1'))