import os
import tarfile
import zipfile
from breezy import osutils, tests
from breezy.errors import UnsupportedOperation
from breezy.export import export
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_tree import TestCaseWithTree
class TestDir(ExportTest, TestCaseWithTree):
    exporter = 'dir'

    def get_export_names(self):
        ret = []
        for dirpath, dirnames, filenames in os.walk('output'):
            for dirname in dirnames:
                ret.append(osutils.pathjoin(dirpath, dirname))
            for filename in filenames:
                ret.append(osutils.pathjoin(dirpath, filename))
        return ret