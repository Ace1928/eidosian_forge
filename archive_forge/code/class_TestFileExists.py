import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
class TestFileExists(TestCase, PathHelpers):

    def test_exists(self):
        tempdir = self.mkdtemp()
        filename = os.path.join(tempdir, 'filename')
        self.touch(filename)
        self.assertThat(filename, FileExists())

    def test_not_exists(self):
        doesntexist = os.path.join(self.mkdtemp(), 'doesntexist')
        mismatch = FileExists().match(doesntexist)
        self.assertThat(PathExists().match(doesntexist).describe(), Equals(mismatch.describe()))

    def test_not_a_file(self):
        tempdir = self.mkdtemp()
        mismatch = FileExists().match(tempdir)
        self.assertThat('%s is not a file.' % tempdir, Equals(mismatch.describe()))