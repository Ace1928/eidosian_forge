import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
class PathHelpers:

    def mkdtemp(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        return directory

    def create_file(self, filename, contents=''):
        fp = open(filename, 'w')
        try:
            fp.write(contents)
        finally:
            fp.close()

    def touch(self, filename):
        return self.create_file(filename)