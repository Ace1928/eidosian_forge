import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
class TestTarballContains(TestCase, PathHelpers):

    def test_match(self):
        tempdir = self.mkdtemp()

        def in_temp_dir(x):
            return os.path.join(tempdir, x)
        self.touch(in_temp_dir('a'))
        self.touch(in_temp_dir('b'))
        tarball = tarfile.open(in_temp_dir('foo.tar.gz'), 'w')
        tarball.add(in_temp_dir('a'), 'a')
        tarball.add(in_temp_dir('b'), 'b')
        tarball.close()
        self.assertThat(in_temp_dir('foo.tar.gz'), TarballContains(['b', 'a']))

    def test_mismatch(self):
        tempdir = self.mkdtemp()

        def in_temp_dir(x):
            return os.path.join(tempdir, x)
        self.touch(in_temp_dir('a'))
        self.touch(in_temp_dir('b'))
        tarball = tarfile.open(in_temp_dir('foo.tar.gz'), 'w')
        tarball.add(in_temp_dir('a'), 'a')
        tarball.add(in_temp_dir('b'), 'b')
        tarball.close()
        mismatch = TarballContains(['d', 'c']).match(in_temp_dir('foo.tar.gz'))
        self.assertEqual(mismatch.describe(), Equals(['c', 'd']).match(['a', 'b']).describe())