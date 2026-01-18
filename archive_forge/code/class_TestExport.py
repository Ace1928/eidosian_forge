import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
class TestExport(TestCaseWithTransport):
    _file_content = b'!\r\n\t\n \r' + b'r29trp9i1r8k0e24c2o7mcx2isrlaw7toh1hv2mtst3o1udkl36v9xn2z8kt\ntvjn7e3i9cj1qs1rw9gcye9w72cbdueiufw8nky7bs08lwggir59d62knecp\n7s0537r8sg3e8hdnidji49rswo47c3j8190nh8emef2b6j1mf5kdq45nt3f5\n1sz9u7fuzrm4w8bebre7p62sh5os2nkj2iiyuk9n0w0pjpdulu9k2aajejah\nini90ny40qzs12ajuy0ua6l178n93lvy2atqngnntsmtlmqx7yhp0q9a1xr4\n1n69kgbo6qu9osjpqq83446r00jijtcstzybfqwm1lnt9spnri2j07bt7bbh\nrf3ejatdxta83te2s0pt9rc4hidgy3d2pc53p4wscdt2b1dfxdj9utf5m17f\nf03oofcau950o090vyx6m72vfkywo7gp3ajzi6uk02dwqwtumq4r44xx6ho7\nnhynborjdjep5j53f9548msb7gd3x9a1xveb4s8zfo6cbdw2kdngcrbakwu8\nql5a8l94gplkwr7oypw5nt1gj5i3xwadyjfr3lb61tfkz31ba7uda9knb294\n1azhfta0q3ry9x36lxyanvhp0g5z0t5a0i4wnoc8p4htexi915y1cnw4nznn\naj70dvp88ifiblv2bsp98hz570teinj8g472ddxni9ydmazfzwtznbf3hrg6\n84gigirjt6n2yagf70036m8d73cz0jpcighpjtxsmbgzbxx7nb4ewq6jbgnc\nhux1b0qtsdi0zfhj6g1otf5jcldmtdvuon8y1ttszkqw3ograwi25yl921hy\nizgscmfha9xdhxxabs07b40secpw22ah9iwpbmsns6qz0yr6fswto3ft2ez5\nngn48pdfxj1pw246drmj1y2ll5af5w7cz849rapzd9ih7qvalw358co0yzrs\nxan9291d1ivjku4o5gjrsnmllrqwxwy86pcivinbmlnzasa9v3o22lgv4uyd\nq8kw77bge3hr5rr5kzwjxk223bkmo3z9oju0954undsz8axr3kb3730otrcr\n9cwhu37htnizdwxmpoc5qmobycfm7ubbykfumv6zgkl6b8zlslwl7a8b81vz\n3weqkvv5csfza9xvwypr6lo0t03fwp0ihmci3m1muh0lf2u30ze0hjag691j\n27fjtd3e3zbiin5n2hq21iuo09ukbs73r5rt7vaw6axvoilvdciir9ugjh2c\nna2b8dr0ptftoyhyxv1iwg661y338e28fhz4xxwgv3hnoe98ydfa1oou45vj\nln74oac2keqt0agbylrqhfscin7ireae2bql7z2le823ksy47ud57z8ctomp\n31s1vwbczdjwqp0o2jc7mkrurvzg8mj2zwcn2iily4gcl4sy4fsh4rignlyz\n'

    def make_basic_tree(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/a', self._file_content)])
        tree.add('a')
        tree.commit('1')
        return tree

    def make_tree_with_extra_bzr_files(self):
        tree = self.make_basic_tree()
        self.build_tree_contents([('tree/.bzrrules', b'')])
        self.build_tree(['tree/.bzr-adir/', 'tree/.bzr-adir/afile'])
        tree.add(['.bzrrules', '.bzr-adir/', '.bzr-adir/afile'])
        self.run_bzr('ignore something -d tree')
        tree.commit('2')
        return tree

    def test_tar_export_ignores_bzr(self):
        tree = self.make_tree_with_extra_bzr_files()
        self.assertTrue(tree.has_filename('.bzrignore'))
        self.assertTrue(tree.has_filename('.bzrrules'))
        self.assertTrue(tree.has_filename('.bzr-adir'))
        self.assertTrue(tree.has_filename('.bzr-adir/afile'))
        self.run_bzr('export test.tar.gz -d tree')
        with tarfile.open('test.tar.gz') as ball:
            self.assertEqual(['test/a'], sorted(ball.getnames()))

    def test_tar_export_unicode_filename(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('tar')
        fname = '€.txt'
        self.build_tree(['tar/' + fname])
        tree.add([fname])
        tree.commit('first')
        self.run_bzr('export test.tar -d tar')
        with tarfile.open('test.tar') as ball:
            self.assertEqual(['test/' + fname], [osutils.safe_unicode(n) for n in ball.getnames()])

    def test_tar_export_unicode_basedir(self):
        """Test for bug #413406"""
        self.requireFeature(features.UnicodeFilenameFeature)
        basedir = '€'
        os.mkdir(basedir)
        self.run_bzr(['init', basedir])
        self.run_bzr(['export', '--format', 'tgz', 'test.tar.gz', '-d', basedir])

    def test_zip_export_ignores_bzr(self):
        tree = self.make_tree_with_extra_bzr_files()
        self.assertTrue(tree.has_filename('.bzrignore'))
        self.assertTrue(tree.has_filename('.bzrrules'))
        self.assertTrue(tree.has_filename('.bzr-adir'))
        self.assertTrue(tree.has_filename('.bzr-adir/afile'))
        self.run_bzr('export test.zip -d tree')
        zfile = zipfile.ZipFile('test.zip')
        self.assertEqual(['test/a'], sorted(zfile.namelist()))

    def assertZipANameAndContent(self, zfile, root=''):
        """The file should only contain name 'a' and _file_content"""
        fname = root + 'a'
        self.assertEqual([fname], sorted(zfile.namelist()))
        zfile.testzip()
        self.assertEqualDiff(self._file_content, zfile.read(fname))

    def test_zip_export_stdout(self):
        tree = self.make_basic_tree()
        contents = self.run_bzr_raw('export -d tree --format=zip -')[0]
        self.assertZipANameAndContent(zipfile.ZipFile(BytesIO(contents)))

    def test_zip_export_file(self):
        tree = self.make_basic_tree()
        self.run_bzr('export -d tree test.zip')
        self.assertZipANameAndContent(zipfile.ZipFile('test.zip'), root='test/')

    def assertTarANameAndContent(self, ball, root=''):
        fname = root + 'a'
        ball_iter = iter(ball)
        tar_info = next(ball_iter)
        self.assertEqual(fname, tar_info.name)
        self.assertEqual(tarfile.REGTYPE, tar_info.type)
        self.assertEqual(len(self._file_content), tar_info.size)
        f = ball.extractfile(tar_info)
        if self._file_content != f.read():
            self.fail('File content has been corrupted. Check that all streams are handled in binary mode.')
        self.assertRaises(StopIteration, next, ball_iter)

    def run_tar_export_disk_and_stdout(self, extension, tarfile_flags):
        tree = self.make_basic_tree()
        fname = 'test.{}'.format(extension)
        self.run_bzr('export -d tree {}'.format(fname))
        mode = 'r|{}'.format(tarfile_flags)
        with tarfile.open(fname, mode=mode) as ball:
            self.assertTarANameAndContent(ball, root='test/')
        content = self.run_bzr_raw('export -d tree --format={} -'.format(extension))[0]
        with tarfile.open(mode=mode, fileobj=BytesIO(content)) as ball:
            self.assertTarANameAndContent(ball, root='')

    def test_tar_export(self):
        self.run_tar_export_disk_and_stdout('tar', '')

    def test_tgz_export(self):
        self.run_tar_export_disk_and_stdout('tgz', 'gz')

    def test_tbz2_export(self):
        self.run_tar_export_disk_and_stdout('tbz2', 'bz2')

    def test_zip_export_unicode(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('zip')
        fname = '€.txt'
        self.build_tree(['zip/' + fname])
        tree.add([fname])
        tree.commit('first')
        os.chdir('zip')
        self.run_bzr('export test.zip')
        zfile = zipfile.ZipFile('test.zip')
        self.assertEqual(['test/' + fname], sorted(zfile.namelist()))

    def test_zip_export_directories(self):
        tree = self.make_branch_and_tree('zip')
        self.build_tree(['zip/a', 'zip/b/', 'zip/b/c', 'zip/d/'])
        tree.add(['a', 'b', 'b/c', 'd'])
        tree.commit('init')
        os.chdir('zip')
        self.run_bzr('export test.zip')
        zfile = zipfile.ZipFile('test.zip')
        names = sorted(zfile.namelist())
        self.assertEqual(['test/a', 'test/b/', 'test/b/c', 'test/d/'], names)
        file_attr = stat.S_IFREG | zip.FILE_PERMISSIONS
        dir_attr = stat.S_IFDIR | zip.ZIP_DIRECTORY_BIT | zip.DIR_PERMISSIONS
        a_info = zfile.getinfo(names[0])
        self.assertEqual(file_attr, a_info.external_attr)
        b_info = zfile.getinfo(names[1])
        self.assertEqual(dir_attr, b_info.external_attr)
        c_info = zfile.getinfo(names[2])
        self.assertEqual(file_attr, c_info.external_attr)
        d_info = zfile.getinfo(names[3])
        self.assertEqual(dir_attr, d_info.external_attr)

    def test_dir_export_nested(self):
        tree = self.make_branch_and_tree('dir')
        self.build_tree(['dir/a'])
        tree.add('a')
        subtree = self.make_branch_and_tree('dir/subdir')
        tree.add_reference(subtree)
        self.build_tree(['dir/subdir/b'])
        subtree.add('b')
        self.run_bzr('export --uncommitted direxport1 dir')
        self.assertFalse(os.path.exists('direxport1/subdir/b'))
        self.run_bzr('export --recurse-nested --uncommitted direxport2 dir')
        self.assertTrue(os.path.exists('direxport2/subdir/b'))

    def test_dir_export(self):
        tree = self.make_branch_and_tree('dir')
        self.build_tree(['dir/a'])
        tree.add('a')
        self.build_tree_contents([('dir/.bzrrules', b'')])
        tree.add('.bzrrules')
        self.build_tree(['dir/.bzr-adir/', 'dir/.bzr-adir/afile'])
        tree.add(['.bzr-adir/', '.bzr-adir/afile'])
        os.chdir('dir')
        self.run_bzr('ignore something')
        tree.commit('1')
        self.assertTrue(tree.has_filename('.bzrignore'))
        self.assertTrue(tree.has_filename('.bzrrules'))
        self.assertTrue(tree.has_filename('.bzr-adir'))
        self.assertTrue(tree.has_filename('.bzr-adir/afile'))
        self.run_bzr('export direxport')
        files = sorted(os.listdir('direxport'))
        self.assertEqual(['a'], files)

    def example_branch(self):
        """Create a branch a 'branch' containing hello and goodbye."""
        tree = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/hello', b'foo')])
        tree.add('hello')
        tree.commit('setup')
        self.build_tree_contents([('branch/goodbye', b'baz')])
        tree.add('goodbye')
        tree.commit('setup')
        return tree

    def test_basic_directory_export(self):
        self.example_branch()
        os.chdir('branch')
        self.run_bzr('export ../latest')
        self.assertEqual(['goodbye', 'hello'], sorted(os.listdir('../latest')))
        self.check_file_contents('../latest/goodbye', b'baz')
        self.run_bzr('export ../first -r 1')
        self.assertEqual(['hello'], sorted(os.listdir('../first')))
        self.check_file_contents('../first/hello', b'foo')
        self.run_bzr('export ../first.gz -r 1')
        self.check_file_contents('../first.gz/hello', b'foo')
        self.run_bzr('export ../first.bz2 -r 1')
        self.check_file_contents('../first.bz2/hello', b'foo')

    def test_basic_tarfile_export(self):
        self.example_branch()
        os.chdir('branch')
        self.run_bzr('export ../first.tar -r 1')
        self.assertTrue(os.path.isfile('../first.tar'))
        with tarfile.open('../first.tar') as tf:
            self.assertEqual(['first/hello'], sorted(tf.getnames()))
            self.assertEqual(b'foo', tf.extractfile('first/hello').read())
        self.run_bzr('export ../first.tar.gz -r 1')
        self.assertTrue(os.path.isfile('../first.tar.gz'))
        self.run_bzr('export ../first.tbz2 -r 1')
        self.assertTrue(os.path.isfile('../first.tbz2'))
        self.run_bzr('export ../first.tar.bz2 -r 1')
        self.assertTrue(os.path.isfile('../first.tar.bz2'))
        self.run_bzr('export ../first.tar.tbz2 -r 1')
        self.assertTrue(os.path.isfile('../first.tar.tbz2'))
        with tarfile.open('../first.tar.tbz2', 'r:bz2') as tf:
            self.assertEqual(['first.tar/hello'], sorted(tf.getnames()))
            self.assertEqual(b'foo', tf.extractfile('first.tar/hello').read())
        self.run_bzr('export ../first2.tar -r 1 --root pizza')
        with tarfile.open('../first2.tar') as tf:
            self.assertEqual(['pizza/hello'], sorted(tf.getnames()))
            self.assertEqual(b'foo', tf.extractfile('pizza/hello').read())

    def test_basic_zipfile_export(self):
        self.example_branch()
        os.chdir('branch')
        self.run_bzr('export ../first.zip -r 1')
        self.assertPathExists('../first.zip')
        with zipfile.ZipFile('../first.zip') as zf:
            self.assertEqual(['first/hello'], sorted(zf.namelist()))
            self.assertEqual(b'foo', zf.read('first/hello'))
        self.run_bzr('export ../first2.zip -r 1 --root pizza')
        with zipfile.ZipFile('../first2.zip') as zf:
            self.assertEqual(['pizza/hello'], sorted(zf.namelist()))
            self.assertEqual(b'foo', zf.read('pizza/hello'))
        self.run_bzr('export ../first-zip --format=zip -r 1')
        with zipfile.ZipFile('../first-zip') as zf:
            self.assertEqual(['first-zip/hello'], sorted(zf.namelist()))
            self.assertEqual(b'foo', zf.read('first-zip/hello'))

    def test_export_from_outside_branch(self):
        self.example_branch()
        self.run_bzr('export latest branch')
        self.assertEqual(['goodbye', 'hello'], sorted(os.listdir('latest')))
        self.check_file_contents('latest/goodbye', b'baz')
        self.run_bzr('export first -r 1 branch')
        self.assertEqual(['hello'], sorted(os.listdir('first')))
        self.check_file_contents('first/hello', b'foo')

    def test_export_partial_tree(self):
        tree = self.example_branch()
        self.build_tree(['branch/subdir/', 'branch/subdir/foo.txt'])
        tree.smart_add(['branch'])
        tree.commit('more setup')
        out, err = self.run_bzr('export exported branch/subdir')
        self.assertEqual(['foo.txt'], os.listdir('exported'))

    def test_dir_export_per_file_timestamps(self):
        tree = self.example_branch()
        self.build_tree_contents([('branch/har', b'foo')])
        tree.add('har')
        tree.commit('setup', timestamp=315532800)
        self.run_bzr('export --per-file-timestamps t branch')
        har_st = os.stat('t/har')
        self.assertEqual(315532800, har_st.st_mtime)

    def test_dir_export_partial_tree_per_file_timestamps(self):
        tree = self.example_branch()
        self.build_tree(['branch/subdir/', 'branch/subdir/foo.txt'])
        tree.smart_add(['branch'])
        tree.commit('setup', timestamp=315532800)
        self.run_bzr('export --per-file-timestamps tpart branch/subdir')
        foo_st = os.stat('tpart/foo.txt')
        self.assertEqual(315532800, foo_st.st_mtime)

    def test_export_directory(self):
        """Test --directory option"""
        self.example_branch()
        self.run_bzr(['export', '--directory=branch', 'latest'])
        self.assertEqual(['goodbye', 'hello'], sorted(os.listdir('latest')))
        self.check_file_contents('latest/goodbye', b'baz')

    def test_export_uncommitted(self):
        """Test --uncommitted option"""
        self.example_branch()
        os.chdir('branch')
        self.build_tree_contents([('goodbye', b'uncommitted data')])
        self.run_bzr(['export', '--uncommitted', 'latest'])
        self.check_file_contents('latest/goodbye', b'uncommitted data')

    def test_export_uncommitted_no_tree(self):
        """Test --uncommitted option only works with a working tree."""
        tree = self.example_branch()
        tree.controldir.destroy_workingtree()
        os.chdir('branch')
        self.run_bzr_error(['brz: ERROR: --uncommitted requires a working tree'], 'export --uncommitted latest')

    def test_zip_export_per_file_timestamps(self):
        tree = self.example_branch()
        self.build_tree_contents([('branch/har', b'foo')])
        tree.add('har')
        timestamp = 347151600
        tree.commit('setup', timestamp=timestamp)
        self.run_bzr('export --per-file-timestamps test.zip branch')
        zfile = zipfile.ZipFile('test.zip')
        info = zfile.getinfo('test/har')
        self.assertEqual(time.localtime(timestamp)[:6], info.date_time)