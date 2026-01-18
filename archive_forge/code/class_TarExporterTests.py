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
class TarExporterTests(tests.TestCaseWithTransport):

    def test_xz(self):
        self.requireFeature(features.lzma)
        import lzma
        wt = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        wt.add(['a'])
        wt.commit('1')
        export.export(wt, 'target.tar.xz', format='txz')
        tf = tarfile.open(fileobj=lzma.LZMAFile('target.tar.xz'))
        self.assertEqual(['target/a'], tf.getnames())

    def test_lzma(self):
        self.requireFeature(features.lzma)
        import lzma
        wt = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        wt.add(['a'])
        wt.commit('1')
        export.export(wt, 'target.tar.lzma', format='tlzma')
        tf = tarfile.open(fileobj=lzma.LZMAFile('target.tar.lzma'))
        self.assertEqual(['target/a'], tf.getnames())

    def test_tgz(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        wt.add(['a'])
        wt.commit('1')
        export.export(wt, 'target.tar.gz', format='tgz')
        tf = tarfile.open('target.tar.gz')
        self.assertEqual(['target/a'], tf.getnames())

    def test_tgz_consistent_mtime(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        wt.add(['a'])
        timestamp = 1547400500
        revid = wt.commit('1', timestamp=timestamp)
        revtree = wt.branch.repository.revision_tree(revid)
        export.export(revtree, 'target.tar.gz', format='tgz')
        with gzip.GzipFile('target.tar.gz', 'r') as f:
            f.read()
            self.assertEqual(int(f.mtime), timestamp)

    def test_tgz_ignores_dest_path(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        wt.add(['a'])
        wt.commit('1')
        os.mkdir('testdir1')
        os.mkdir('testdir2')
        export.export(wt, 'testdir1/target.tar.gz', format='tgz', per_file_timestamps=True)
        export.export(wt, 'testdir2/target.tar.gz', format='tgz', per_file_timestamps=True)
        file1 = open('testdir1/target.tar.gz', 'rb')
        self.addCleanup(file1.close)
        file2 = open('testdir1/target.tar.gz', 'rb')
        self.addCleanup(file2.close)
        content1 = file1.read()
        content2 = file2.read()
        self.assertEqualDiff(content1, content2)
        self.assertFalse(b'testdir1' in content1)
        self.assertFalse(b'target.tar.gz' in content1)
        self.assertTrue(b'target.tar' in content1)

    def test_tbz2(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        wt.add(['a'])
        wt.commit('1')
        export.export(wt, 'target.tar.bz2', format='tbz2')
        tf = tarfile.open('target.tar.bz2')
        self.assertEqual(['target/a'], tf.getnames())

    def test_export_tarball_generator(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        wt.add(['a'])
        wt.commit('1', timestamp=42)
        target = BytesIO()
        with wt.lock_read():
            target.writelines(tarball_generator(wt, 'bar'))
        target.seek(0)
        ball2 = tarfile.open(None, 'r', target)
        self.addCleanup(ball2.close)
        self.assertEqual(['bar/a'], ball2.getnames())