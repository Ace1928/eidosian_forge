import gzip
import os
import tarfile
import time
import zipfile
from io import BytesIO
from .. import errors, export, tests
from ..archive.tar import tarball_generator
from ..export import get_root_name
from . import features
class ZipExporterTests(tests.TestCaseWithTransport):

    def test_per_file_timestamps(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('har', b'foo')])
        tree.add('har')
        timestamp = 347151600
        tree.commit('setup', timestamp=timestamp)
        export.export(tree.basis_tree(), 'test.zip', format='zip', per_file_timestamps=True)
        zfile = zipfile.ZipFile('test.zip')
        info = zfile.getinfo('test/har')
        self.assertEqual(time.localtime(timestamp)[:6], info.date_time)