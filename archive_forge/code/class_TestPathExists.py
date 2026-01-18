import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
class TestPathExists(TestCase, PathHelpers):

    def test_exists(self):
        tempdir = self.mkdtemp()
        self.assertThat(tempdir, PathExists())

    def test_not_exists(self):
        doesntexist = os.path.join(self.mkdtemp(), 'doesntexist')
        mismatch = PathExists().match(doesntexist)
        self.assertThat('%s does not exist.' % doesntexist, Equals(mismatch.describe()))