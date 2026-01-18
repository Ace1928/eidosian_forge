import os
import tarfile
import zipfile
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.per_tree import TestCaseWithTree
class TestTgz(ArchiveTests, TestCaseWithTree):
    format = 'tgz'

    def get_export_names(self, path):
        tf = tarfile.open(path)
        try:
            return tf.getnames()
        finally:
            tf.close()