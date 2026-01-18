import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
class TestDirExists(TestCase, PathHelpers):

    def test_exists(self):
        tempdir = self.mkdtemp()
        self.assertThat(tempdir, DirExists())

    def test_not_exists(self):
        doesntexist = os.path.join(self.mkdtemp(), 'doesntexist')
        mismatch = DirExists().match(doesntexist)
        self.assertThat(PathExists().match(doesntexist).describe(), Equals(mismatch.describe()))

    def test_not_a_directory(self):
        filename = os.path.join(self.mkdtemp(), 'foo')
        self.touch(filename)
        mismatch = DirExists().match(filename)
        self.assertThat('%s is not a directory.' % filename, Equals(mismatch.describe()))