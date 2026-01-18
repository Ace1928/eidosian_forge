from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gzip
import os
import six
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.exception import NO_URLS_MATCHED_GENERIC
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetDummyProjectForUnitTest
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils import copy_helper
from gslib.utils import system_util
class GsutilNamingTests(testcase.GsUtilUnitTestCase):
    """Unit tests for gsutil naming logic."""

    def testGetPathBeforeFinalDir(self):
        """Tests GetPathBeforeFinalDir() (unit test)."""
        self.assertEqual('gs://', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/'), StorageUrlFromString('gs://bucket/obj')))
        self.assertEqual('gs://bucket', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/dir/'), StorageUrlFromString('gs://bucket/dir/obj')))
        self.assertEqual('gs://bucket', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/dir'), StorageUrlFromString('gs://bucket/dir/obj')))
        self.assertEqual('gs://bucket/dir', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/dir/obj'), StorageUrlFromString('gs://bucket/dir/obj')))
        self.assertEqual('gs://bucket/dir1', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/*/dir2'), StorageUrlFromString('gs://bucket/dir1/dir2/obj')))
        self.assertEqual('gs://bucket/dir1/dir2/dir3', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/*/dir2/*/dir4'), StorageUrlFromString('gs://bucket/dir1/dir2/dir3/dir4/obj')))
        src_dir = self.CreateTempDir()
        subdir = os.path.join(src_dir, 'subdir')
        os.mkdir(subdir)
        self.assertEqual(suri(src_dir), copy_helper.GetPathBeforeFinalDir(StorageUrlFromString(suri(subdir)), StorageUrlFromString(suri(subdir, 'obj'))))

    def testCopyingTopLevelFileToBucket(self):
        """Tests copying one top-level file to a bucket."""
        src_file = self.CreateTempFile(file_name='f0')
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', [src_file, suri(dst_bucket_uri)])
        actual = list(self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True))
        self.assertEqual(1, len(actual))
        self.assertEqual('f0', actual[0].root_object.name)

    def testCopyingMultipleFilesToBucket(self):
        """Tests copying multiple files to a bucket."""
        src_file0 = self.CreateTempFile(file_name='f0')
        src_file1 = self.CreateTempFile(file_name='f1')
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', [src_file0, src_file1, suri(dst_bucket_uri)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'f0'), suri(dst_bucket_uri, 'f1')])
        self.assertEqual(expected, actual)

    def testCopyingNestedFileToBucketSubdir(self):
        """Tests copying a nested file to a bucket subdir.

    Tests that we correctly translate local FS-specific delimiters (\\ on
    Windows) to bucket delimiter (/).
    """
        tmpdir = self.CreateTempDir()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        src_file = self.CreateTempFile(tmpdir=tmpdir, file_name='obj', contents=b'')
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', [src_file, suri(dst_bucket_uri, 'subdir/a')])
        self.RunCommand('cp', [src_file, suri(dst_bucket_uri, 'subdir')])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterObjects()))
        expected = set([suri(dst_bucket_uri, 'subdir', 'a'), suri(dst_bucket_uri, 'subdir', 'obj')])
        self.assertEqual(expected, actual)

    def testCopyingBucketSubdirsToBucket(self):
        """Ensure wildcarded recursive cp in bucket subdirs behaves like Unix."""
        src_bucket_uri = self.CreateBucket()
        dst_bucket_uri = self.CreateBucket()
        fpath = self.CreateTempFile(file_name='foo', contents=b'bar')
        self.RunCommand('cp', [fpath, suri(src_bucket_uri, 'Test/sub-test/foo')])
        self.RunCommand('cp', [fpath, suri(src_bucket_uri, 'Test2/sub-test/foo')])
        self.RunCommand('cp', [fpath, suri(src_bucket_uri, 'Test3/sub-test/foo')])
        self.RunCommand('cp', ['-R', suri(src_bucket_uri, '*', 'sub-test'), suri(dst_bucket_uri)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'sub-test', 'foo')])
        self.assertEqual(expected, actual)
        src_bucket_uri2 = self.CreateBucket()
        dst_bucket_uri2 = self.CreateBucket()
        self.RunCommand('cp', [fpath, suri(src_bucket_uri2, 'Test/dir1/dir2/foo')])
        self.RunCommand('cp', [fpath, suri(src_bucket_uri2, 'Test2/dir1/dir2/foo')])
        self.RunCommand('cp', [fpath, suri(src_bucket_uri2, 'Test3/dir1/dir2/bar')])
        self.RunCommand('cp', ['-R', suri(src_bucket_uri2, '*', 'dir1', 'dir2'), suri(dst_bucket_uri2)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri2, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri2, 'dir2', 'foo'), suri(dst_bucket_uri2, 'dir2', 'bar')])
        self.assertEqual(expected, actual)
        dst_bucket_uri3 = self.CreateBucket()
        self.RunCommand('cp', ['-R', suri(src_bucket_uri2, 'Test*', '*', 'dir2'), suri(dst_bucket_uri3)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri3, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri3, 'dir2', 'foo'), suri(dst_bucket_uri3, 'dir2', 'bar')])
        self.assertEqual(expected, actual)
        src_bucket_uri3 = self.CreateBucket()
        dst_bucket_uri4 = self.CreateBucket()
        self.RunCommand('cp', [fpath, suri(src_bucket_uri3, 'dir1/test1/dir2/dir3/foo')])
        self.RunCommand('cp', [fpath, suri(src_bucket_uri3, 'dir1/test2/dir2/dir3/foo')])
        self.RunCommand('cp', [fpath, suri(src_bucket_uri3, 'dir1/test3/dir2/dir3/bar')])
        self.RunCommand('cp', ['-R', suri(src_bucket_uri3, 'dir1', '*', 'dir2', 'dir3'), suri(dst_bucket_uri4)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri4, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri4, 'dir3', 'foo'), suri(dst_bucket_uri4, 'dir3', 'bar')])
        self.assertEqual(expected, actual)

    def testCopyingAbsolutePathDirToBucket(self):
        """Tests recursively copying absolute path directory to a bucket."""
        dst_bucket_uri = self.CreateBucket()
        src_dir_root = self.CreateTempDir(test_files=['f0', 'f1', 'f2.txt', ('dir0', 'dir1', 'nested')])
        self.RunCommand('cp', ['-R', src_dir_root, suri(dst_bucket_uri)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        src_tmpdir = os.path.split(src_dir_root)[1]
        expected = set([suri(dst_bucket_uri, src_tmpdir, 'f0'), suri(dst_bucket_uri, src_tmpdir, 'f1'), suri(dst_bucket_uri, src_tmpdir, 'f2.txt'), suri(dst_bucket_uri, src_tmpdir, 'dir0', 'dir1', 'nested')])
        self.assertEqual(expected, actual)

    def testCopyingRelativePathDirToBucket(self):
        """Tests recursively copying relative directory to a bucket."""
        dst_bucket_uri = self.CreateBucket()
        src_dir = self.CreateTempDir(test_files=[('dir0', 'f1')])
        self.RunCommand('cp', ['-R', 'dir0', suri(dst_bucket_uri)], cwd=src_dir)
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dir0', 'f1')])
        self.assertEqual(expected, actual)

    def testCopyingRelPathSubDirToBucketSubdirWithDollarFolderObj(self):
        """Tests recursively copying relative sub-directory to bucket subdir.

    Subdir is signified by a $folder$ object.
    """
        dst_bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=dst_bucket_uri, object_name='abc_$folder$', contents='')
        src_dir = self.CreateTempDir(test_files=[('dir0', 'dir1', 'f1')])
        self.RunCommand('cp', ['-R', os.path.join('dir0', 'dir1'), suri(dst_bucket_uri, 'abc')], cwd=src_dir)
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'abc_$folder$'), suri(dst_bucket_uri, 'abc', 'dir1', 'f1')])
        self.assertEqual(expected, actual)

    def testCopyingRelativePathSubDirToBucketSubdirSignifiedBySlash(self):
        """Tests recursively copying relative sub-directory to bucket subdir.

    Subdir is signified by a / object.
    """
        dst_bucket_uri = self.CreateBucket()
        src_dir = self.CreateTempDir(test_files=[('dir0', 'dir1', 'f1')])
        self.RunCommand('cp', ['-R', os.path.join('dir0', 'dir1'), suri(dst_bucket_uri, 'abc') + '/'], cwd=src_dir)
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'abc', 'dir1', 'f1')])
        self.assertEqual(expected, actual)

    def testCopyingRelativePathSubDirToBucket(self):
        """Tests recursively copying relative sub-directory to a bucket."""
        dst_bucket_uri = self.CreateBucket()
        src_dir = self.CreateTempDir(test_files=[('dir0', 'dir1', 'f1')])
        self.RunCommand('cp', ['-R', os.path.join('dir0', 'dir1'), suri(dst_bucket_uri)], cwd=src_dir)
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dir1', 'f1')])
        self.assertEqual(expected, actual)

    def testCopyingDotSlashToBucket(self):
        """Tests copying ./ to a bucket produces expected naming."""
        dst_bucket_uri = self.CreateBucket()
        src_dir = self.CreateTempDir(test_files=['foo'])
        for rel_src_dir in ['.', '.%s' % os.sep]:
            self.RunCommand('cp', ['-R', rel_src_dir, suri(dst_bucket_uri)], cwd=src_dir)
            actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
            expected = set([suri(dst_bucket_uri, 'foo')])
            self.assertEqual(expected, actual)

    def testCopyingDirContainingOneFileToBucket(self):
        """Tests copying a directory containing 1 file to a bucket.

    We test this case to ensure that correct bucket handling isn't dependent
    on the copy being treated as a multi-source copy.
    """
        dst_bucket_uri = self.CreateBucket()
        src_dir = self.CreateTempDir(test_files=[('dir0', 'dir1', 'foo')])
        self.RunCommand('cp', ['-R', os.path.join(src_dir, 'dir0', 'dir1'), suri(dst_bucket_uri)])
        actual = list((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(1, len(actual))
        self.assertEqual(suri(dst_bucket_uri, 'dir1', 'foo'), actual[0])

    def testCopyDotFilesToBucket(self):
        dst_bucket_uri = self.CreateBucket()
        src_dir = self.CreateTempDir(test_files=['foo'])
        object_named_dot = suri(dst_bucket_uri) + '/.'
        object_named_dotdot = suri(dst_bucket_uri) + '/..'
        for object_name in (object_named_dot, object_named_dotdot):
            try:
                self.RunCommand('cp', [os.path.join(src_dir, 'foo'), object_name])
                self.fail('Expected InvalidUrlError for %s' % object_name)
            except InvalidUrlError:
                pass

    def testCopyingBucketToDir(self):
        """Tests copying from a bucket to a directory."""
        src_bucket_uri = self.CreateBucket(test_objects=['foo', 'dir/foo2'])
        dst_dir = self.CreateTempDir()
        with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
            self.RunCommand('cp', ['-R', suri(src_bucket_uri), dst_dir])
        actual = set((str(u) for u in self._test_wildcard_iterator('%s%s**' % (dst_dir, os.sep)).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_dir, src_bucket_uri.bucket_name, 'foo'), suri(dst_dir, src_bucket_uri.bucket_name, 'dir', 'foo2')])
        self.assertEqual(expected, actual)

    @unittest.skipIf(system_util.IS_WINDOWS, 'os.symlink() is not available on Windows.')
    def testCopyingSymlinkDirectory(self):
        """Tests that cp warns when copying a symlink directory."""
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        tmpdir2 = self.CreateTempDir()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        fpath1 = self.CreateTempFile(tmpdir=subdir, contents=b'foo')
        self.CreateTempFile(tmpdir=tmpdir2, contents=b'foo')
        os.mkdir(os.path.join(tmpdir, 'symlinkdir'))
        os.symlink(tmpdir2, os.path.join(subdir, 'symlinkdir'))
        mock_log_handler = self.RunCommand('cp', ['-r', tmpdir, suri(bucket_uri)], debug=1, return_log_handler=True)
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected_object_path = suri(bucket_uri, os.path.basename(tmpdir), 'subdir', os.path.basename(fpath1))
        expected = set([expected_object_path])
        self.assertEqual(expected, actual)
        desired_msg = 'Skipping symlink directory "%s"' % os.path.join(subdir, 'symlinkdir')
        self.assertIn(desired_msg, mock_log_handler.messages['info'], '"%s" not found in mock_log_handler["info"]: %s' % (desired_msg, str(mock_log_handler.messages)))

    def testCopyingBucketToBucket(self):
        """Tests copying from a bucket-only URI to a bucket."""
        src_bucket_uri = self.CreateBucket(test_objects=['foo', 'dir/foo2'])
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', ['-R', suri(src_bucket_uri), suri(dst_bucket_uri)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, src_bucket_uri.bucket_name, 'foo'), suri(dst_bucket_uri, src_bucket_uri.bucket_name, 'dir', 'foo2')])
        self.assertEqual(expected, actual)

    def testCopyingDirectoryToDirectory(self):
        """Tests copying from a directory to a directory."""
        src_dir = self.CreateTempDir(test_files=['foo', ('dir', 'foo2')])
        dst_dir = self.CreateTempDir()
        self.RunCommand('cp', ['-R', src_dir, dst_dir])
        actual = set((str(u) for u in self._test_wildcard_iterator('%s%s**' % (dst_dir, os.sep)).IterAll(expand_top_level_buckets=True)))
        src_dir_base = os.path.split(src_dir)[1]
        expected = set([suri(dst_dir, src_dir_base, 'foo'), suri(dst_dir, src_dir_base, 'dir', 'foo2')])
        self.assertEqual(expected, actual)

    def testCopyingFilesAndDirNonRecursive(self):
        """Tests copying containing files and a directory without -R."""
        src_dir = self.CreateTempDir(test_files=['foo', 'bar', ('d1', 'f2'), ('d2', 'f3'), ('d3', 'd4', 'f4')])
        dst_dir = self.CreateTempDir()
        self.RunCommand('cp', ['%s%s*' % (src_dir, os.sep), dst_dir])
        actual = set((str(u) for u in self._test_wildcard_iterator('%s%s**' % (dst_dir, os.sep)).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_dir, 'foo'), suri(dst_dir, 'bar')])
        self.assertEqual(expected, actual)

    def testCopyingFileToDir(self):
        """Tests copying one file to a directory."""
        src_file = self.CreateTempFile(file_name='foo')
        dst_dir = self.CreateTempDir()
        self.RunCommand('cp', [src_file, dst_dir])
        actual = list(self._test_wildcard_iterator('%s%s*' % (dst_dir, os.sep)).IterAll(expand_top_level_buckets=True))
        self.assertEqual(1, len(actual))
        self.assertEqual(suri(dst_dir, 'foo'), str(actual[0]))

    def testCopyingFileToObjectWithConsecutiveSlashes(self):
        """Tests copying a file to an object containing consecutive slashes."""
        src_file = self.CreateTempFile(file_name='f0')
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', [src_file, suri(dst_bucket_uri) + '//obj'])
        actual = list(self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True))
        self.assertEqual(1, len(actual))
        self.assertEqual('/obj', actual[0].root_object.name)

    def testCopyingCompressedFileToBucket(self):
        """Tests copying one file with compression to a bucket."""
        src_file = self.CreateTempFile(contents=b'plaintext', file_name='f2.txt')
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', ['-z', 'txt', src_file, suri(dst_bucket_uri)])
        actual = list(self._test_wildcard_iterator(suri(dst_bucket_uri, '*')).IterAll(expand_top_level_buckets=True))
        self.assertEqual(1, len(actual))
        actual_obj = actual[0].root_object
        self.assertEqual('f2.txt', actual_obj.name)
        self.assertEqual('gzip', actual_obj.contentEncoding)
        stdout = self.RunCommand('cat', [suri(dst_bucket_uri, 'f2.txt')], return_stdout=True)
        f = gzip.GzipFile(fileobj=six.BytesIO(six.ensure_binary(stdout)), mode='rb')
        try:
            self.assertEqual(f.read(), b'plaintext')
        finally:
            f.close()

    def testCopyingObjectToObject(self):
        """Tests copying an object to an object."""
        src_bucket_uri = self.CreateBucket(test_objects=['obj'])
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', [suri(src_bucket_uri, 'obj'), suri(dst_bucket_uri)])
        actual = list(self._test_wildcard_iterator(suri(dst_bucket_uri, '*')).IterAll(expand_top_level_buckets=True))
        self.assertEqual(1, len(actual))
        self.assertEqual('obj', actual[0].root_object.name)

    def testCopyingObjectToObjectUsingDestWildcard(self):
        """Tests copying an object to an object using a dest wildcard."""
        src_bucket_uri = self.CreateBucket(test_objects=['obj'])
        dst_bucket_uri = self.CreateBucket(test_objects=['dstobj'])
        self.RunCommand('cp', [suri(src_bucket_uri, 'obj'), '%s*' % dst_bucket_uri.uri])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '*')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dstobj')])
        self.assertEqual(expected, actual)

    def testCopyingObjsAndFilesToDir(self):
        """Tests copying objects and files to a directory."""
        src_bucket_uri = self.CreateBucket(test_objects=['f1'])
        src_dir = self.CreateTempDir(test_files=['f2'])
        dst_dir = self.CreateTempDir()
        with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
            self.RunCommand('cp', ['-R', suri(src_bucket_uri, '**'), os.path.join(src_dir, '**'), dst_dir])
        actual = set((str(u) for u in self._test_wildcard_iterator(os.path.join(dst_dir, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_dir, 'f1'), suri(dst_dir, 'f2')])
        self.assertEqual(expected, actual)

    def testCopyingObjToDot(self):
        """Tests that copying an object to . or ./ downloads to correct name."""
        src_bucket_uri = self.CreateBucket(test_objects=['f1'])
        dst_dir = self.CreateTempDir()
        for final_char in ('/', ''):
            with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
                self.RunCommand('cp', [suri(src_bucket_uri, 'f1'), '.%s' % final_char], cwd=dst_dir)
            actual = set()
            for dirname, dirnames, filenames in os.walk(dst_dir):
                for subdirname in dirnames:
                    actual.add(os.path.join(dirname, subdirname))
                for filename in filenames:
                    actual.add(os.path.join(dirname, filename))
            expected = set([os.path.join(dst_dir, 'f1')])
            self.assertEqual(expected, actual)

    def testCopyingObjsAndFilesToBucket(self):
        """Tests copying objects and files to a bucket."""
        src_bucket_uri = self.CreateBucket(test_objects=['f1'])
        src_dir = self.CreateTempDir(test_files=['f2'])
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', ['-R', suri(src_bucket_uri, '**'), '%s%s**' % (src_dir, os.sep), suri(dst_bucket_uri)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'f1'), suri(dst_bucket_uri, 'f2')])
        self.assertEqual(expected, actual)

    def testCopyingSubdirRecursiveToNonexistentSubdir(self):
        """Tests copying a directory with a single file recursively to a bucket.

    The file should end up in a new bucket subdirectory with the file's
    directory structure starting below the recursive copy point, as in Unix cp.

    Example:
      filepath: dir1/dir2/foo
      cp -r dir1 dir3
      Results in dir3/dir2/foo being created.
    """
        src_dir = self.CreateTempDir()
        self.CreateTempFile(tmpdir=src_dir + '/dir1/dir2', file_name='foo')
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', ['-R', src_dir + '/dir1', suri(dst_bucket_uri, 'dir3')])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dir3/dir2/foo')])
        self.assertEqual(expected, actual)

    def testCopyingFileToDirRecursive(self):
        """Tests copying a file with -R."""
        src_file = self.CreateTempFile(file_name='foo')
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', ['-R', src_file, suri(dst_bucket_uri, 'dir/foo')])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dir/foo')])
        self.assertEqual(expected, actual)

    def testCopyingFileToNonExistentDir(self):
        """Tests copying a file to a non-existent directory.

    Should create the directory and add the file to it
    """
        src_file = self.CreateTempFile(file_name='foo')
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', [src_file, suri(dst_bucket_uri, 'dir') + '/'])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dir/foo')])
        self.assertEqual(expected, actual)

    def testCopyingMultipleFilesToDirRecursive(self):
        """Tests copying multiple files with -R."""
        src_dir = self.CreateTempDir()
        src_file1 = self.CreateTempFile(tmpdir=src_dir, file_name='foo')
        src_file2 = self.CreateTempFile(tmpdir=src_dir, file_name='bar')
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', ['-R', src_file1, src_file2, suri(dst_bucket_uri, 'dir/foo')])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dir/foo/foo'), suri(dst_bucket_uri, 'dir/foo/bar')])
        self.assertEqual(expected, actual)

    def testAttemptDirCopyWithoutRecursion(self):
        """Tests copying a directory without -R."""
        src_dir = self.CreateTempDir(test_files=1)
        dst_dir = self.CreateTempDir()
        try:
            self.RunCommand('cp', [src_dir, dst_dir])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertIn(NO_URLS_MATCHED_GENERIC, e.reason)

    def testNonRecursiveFileAndSameNameSubdir(self):
        """Tests copying a file and subdirectory of the same name without -R."""
        src_bucket_uri = self.CreateBucket(test_objects=['f1', 'f1/f2'])
        dst_dir = self.CreateTempDir()
        with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
            self.RunCommand('cp', [suri(src_bucket_uri, 'f1'), dst_dir])
        actual = list(self._test_wildcard_iterator('%s%s*' % (dst_dir, os.sep)).IterAll(expand_top_level_buckets=True))
        self.assertEqual(1, len(actual))
        self.assertEqual(suri(dst_dir, 'f1'), str(actual[0]))

    def testAttemptCopyingProviderOnlySrc(self):
        """Attempts to copy a src specified as a provider-only URI."""
        src_bucket_uri = self.CreateBucket()
        try:
            self.RunCommand('cp', ['gs://', suri(src_bucket_uri)])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertIn('provider-only', e.reason)

    def testAttemptCopyingOverlappingSrcDstFile(self):
        """Attempts to an object atop itself."""
        src_file = self.CreateTempFile()
        try:
            self.RunCommand('cp', [src_file, src_file])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertIn('are the same file - abort', e.reason)

    def testAttemptCopyingToMultiMatchWildcard(self):
        """Attempts to copy where dst wildcard matches >1 obj."""
        src_bucket_uri = self.CreateBucket(test_objects=2)
        try:
            self.RunCommand('cp', [suri(src_bucket_uri, 'obj0'), suri(src_bucket_uri, '*')])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertNotEqual(e.reason.find('must match exactly 1 URL'), -1)

    def testAttemptCopyingMultiObjsToFile(self):
        """Attempts to copy multiple objects to a file."""
        src_bucket_uri = self.CreateBucket(test_objects=2)
        dst_file = self.CreateTempFile()
        try:
            self.RunCommand('cp', ['-R', suri(src_bucket_uri, '*'), dst_file])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertIn('must name a directory, bucket, or', e.reason)

    def testAttemptCopyingWithFileDirConflict(self):
        """Attempts to copy objects that cause a file/directory conflict."""
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name='a')
        self.CreateObject(bucket_uri=bucket_uri, object_name='b/a')
        dst_dir = self.CreateTempDir()
        try:
            self.RunCommand('cp', ['-R', suri(bucket_uri), dst_dir])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertNotEqual('exists where a directory needs to be created', e.reason)

    def testAttemptCopyingWithDirFileConflict(self):
        """Attempts to copy an object that causes a directory/file conflict."""
        tmpdir = self.CreateTempDir()
        os.mkdir(os.path.join(tmpdir, 'abc'))
        src_uri = self.CreateObject(object_name='abc', contents='bar')
        try:
            self.RunCommand('cp', [suri(src_uri), tmpdir + '/'])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertNotEqual('where the file needs to be created', e.reason)

    def testWildcardMoveWithinBucket(self):
        """Attempts to move using src wildcard that overlaps dest object.

    We want to ensure that this doesn't stomp the result data.
    """
        dst_bucket_uri = self.CreateBucket(test_objects=['old'])
        self.RunCommand('mv', [suri(dst_bucket_uri, 'old*'), suri(dst_bucket_uri, 'new')])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'new')])
        self.assertEqual(expected, actual)

    def testLsNonExistentObjectWithPrefixName(self):
        """Test ls of non-existent obj that matches prefix of existing objs."""
        src_bucket_uri = self.CreateBucket(test_objects=['obj_with_suffix'])
        try:
            self.RunCommand('ls', [suri(src_bucket_uri, 'obj')])
        except CommandException as e:
            self.assertIn('matched no objects', e.reason)

    def testLsBucketNonRecursive(self):
        """Test that ls of a bucket returns expected results."""
        src_bucket_uri = self.CreateBucket(test_objects=['foo1', 'd0/foo2', 'd1/d2/foo3'])
        output = self.RunCommand('ls', [suri(src_bucket_uri, '*')], return_stdout=True)
        expected = set([suri(src_bucket_uri, 'foo1'), suri(src_bucket_uri, 'd1', ':'), suri(src_bucket_uri, 'd1', 'd2') + src_bucket_uri.delim, suri(src_bucket_uri, 'd0', ':'), suri(src_bucket_uri, 'd0', 'foo2')])
        expected.add('')
        actual = set([line.strip() for line in output.split('\n')])
        self.assertEqual(expected, actual)

    def testLsBucketRecursive(self):
        """Test that ls -R of a bucket returns expected results."""
        src_bucket_uri = self.CreateBucket(test_objects=['foo1', 'd0/foo2', 'd1/d2/foo3'])
        output = self.RunCommand('ls', ['-R', suri(src_bucket_uri, '*')], return_stdout=True)
        expected = set([suri(src_bucket_uri, 'foo1'), suri(src_bucket_uri, 'd1', ':'), suri(src_bucket_uri, 'd1', 'd2', ':'), suri(src_bucket_uri, 'd1', 'd2', 'foo3'), suri(src_bucket_uri, 'd0', ':'), suri(src_bucket_uri, 'd0', 'foo2')])
        expected.add('')
        actual = set([line.strip() for line in output.split('\n')])
        self.assertEqual(expected, actual)

    def testLsBucketRecursiveWithLeadingSlashObjectName(self):
        """Test that ls -R of a bucket with an object that has leading slash."""
        dst_bucket_uri = self.CreateBucket(test_objects=['f0'])
        output = self.RunCommand('ls', ['-R', suri(dst_bucket_uri, '*')], return_stdout=True)
        expected = set([suri(dst_bucket_uri, 'f0')])
        expected.add('')
        actual = set([line.strip() for line in output.split('\n')])
        self.assertEqual(expected, actual)

    def testLsBucketSubdirNonRecursive(self):
        """Test that ls of a bucket subdir returns expected results."""
        src_bucket_uri = self.CreateBucket(test_objects=['src_subdir/foo', 'src_subdir/nested/foo2'])
        output = self.RunCommand('ls', [suri(src_bucket_uri, 'src_subdir')], return_stdout=True)
        expected = set([suri(src_bucket_uri, 'src_subdir', 'foo'), suri(src_bucket_uri, 'src_subdir', 'nested') + src_bucket_uri.delim])
        expected.add('')
        actual = set([line.strip() for line in output.split('\n')])
        self.assertEqual(expected, actual)

    def testLsBucketSubdirRecursive(self):
        """Test that ls -R of a bucket subdir returns expected results."""
        src_bucket_uri = self.CreateBucket(test_objects=['src_subdir/foo', 'src_subdir/nested/foo2'])
        for final_char in ('/', ''):
            output = self.RunCommand('ls', ['-R', suri(src_bucket_uri, 'src_subdir') + final_char], return_stdout=True)
            expected = set([suri(src_bucket_uri, 'src_subdir', ':'), suri(src_bucket_uri, 'src_subdir', 'foo'), suri(src_bucket_uri, 'src_subdir', 'nested', ':'), suri(src_bucket_uri, 'src_subdir', 'nested', 'foo2')])
            expected.add('')
            actual = set([line.strip() for line in output.split('\n')])
            self.assertEqual(expected, actual)

    def testSetAclOnBucketRuns(self):
        """Test that the 'acl set' command basically runs."""
        src_bucket_uri = self.CreateBucket()
        self.RunCommand('acl', ['set', 'private', suri(src_bucket_uri)])

    def testSetAclOnWildcardNamedBucketRuns(self):
        """Test that 'acl set' basically runs against wildcard-named bucket."""
        src_bucket_uri = self.CreateBucket(test_objects=['f0'])
        with SetDummyProjectForUnitTest():
            self.RunCommand('acl', ['set', 'private', suri(src_bucket_uri)[:-2] + '*'])

    def testSetAclOnObjectRuns(self):
        """Test that the 'acl set' command basically runs."""
        src_bucket_uri = self.CreateBucket(test_objects=['f0'])
        self.RunCommand('acl', ['set', 'private', suri(src_bucket_uri, '*')])

    def testSetDefAclOnBucketRuns(self):
        """Test that the 'defacl set' command basically runs."""
        src_bucket_uri = self.CreateBucket()
        self.RunCommand('defacl', ['set', 'private', suri(src_bucket_uri)])

    def testSetDefAclOnObjectFails(self):
        """Test that the 'defacl set' command fails when run against an object."""
        src_bucket_uri = self.CreateBucket()
        try:
            self.RunCommand('defacl', ['set', 'private', suri(src_bucket_uri, '*')])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertIn('URL must name a bucket', e.reason)

    def testMinusDOptionWorks(self):
        """Tests using gsutil -D option."""
        src_file = self.CreateTempFile(file_name='f0')
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', [src_file, suri(dst_bucket_uri)], debug=3)
        actual = list(self._test_wildcard_iterator(suri(dst_bucket_uri, '*')).IterAll(expand_top_level_buckets=True))
        self.assertEqual(1, len(actual))
        self.assertEqual('f0', actual[0].root_object.name)

    def testFlatCopyingObjsAndFilesToBucketSubDir(self):
        """Tests copying flatly listed objects and files to bucket subdir."""
        src_bucket_uri = self.CreateBucket(test_objects=['f0', 'd0/f1', 'd1/d2/f2'])
        src_dir = self.CreateTempDir(test_files=['f3', ('d3', 'f4'), ('d4', 'd5', 'f5')])
        dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir0/existing', 'dst_subdir1/existing'])
        for i, final_char in enumerate(('/', '')):
            self.RunCommand('cp', ['-R', suri(src_bucket_uri, '**'), os.path.join(src_dir, '**'), suri(dst_bucket_uri, 'dst_subdir%d' % i) + final_char])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set()
        for i in range(2):
            expected.add(suri(dst_bucket_uri, 'dst_subdir%d' % i, 'existing'))
            for j in range(6):
                expected.add(suri(dst_bucket_uri, 'dst_subdir%d' % i, 'f%d' % j))
        self.assertEqual(expected, actual)

    def testRecursiveCopyObjsAndFilesToExistingBucketSubDir(self):
        """Tests recursive copy of objects and files to existing bucket subdir."""
        src_bucket_uri = self.CreateBucket(test_objects=['f0', 'nested/f1'])
        dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir0/existing_obj', 'dst_subdir1/existing_obj'])
        src_dir = self.CreateTempDir(test_files=['f2', ('nested', 'f3')])
        for i, final_char in enumerate(('/', '')):
            self.RunCommand('cp', ['-R', suri(src_bucket_uri), src_dir, suri(dst_bucket_uri, 'dst_subdir%d' % i) + final_char])
            actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, 'dst_subdir%d' % i, '**')).IterAll(expand_top_level_buckets=True)))
            tmp_dirname = os.path.split(src_dir)[1]
            bucketname = src_bucket_uri.bucket_name
            expected = set([suri(dst_bucket_uri, 'dst_subdir%d' % i, 'existing_obj'), suri(dst_bucket_uri, 'dst_subdir%d' % i, bucketname, 'f0'), suri(dst_bucket_uri, 'dst_subdir%d' % i, bucketname, 'nested', 'f1'), suri(dst_bucket_uri, 'dst_subdir%d' % i, tmp_dirname, 'f2'), suri(dst_bucket_uri, 'dst_subdir%d' % i, tmp_dirname, 'nested', 'f3')])
            self.assertEqual(expected, actual)

    def testRecursiveCopyFileToExistingBucketSubDirInvalidSourceParent(self):
        """Tests recursive copy of invalid path file to existing bucket subdir."""
        src_dir = self.CreateTempDir(test_files=[('nested', 'f0')])
        dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir/existing_obj'])
        for relative_path_string in ['.', '.' + os.sep]:
            self.RunCommand('cp', ['-R', src_dir + os.sep + relative_path_string, suri(dst_bucket_uri, 'dst_subdir')])
            actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, 'dst_subdir', '**')).IterAll(expand_top_level_buckets=True)))
            expected = set([suri(dst_bucket_uri, 'dst_subdir', 'nested', 'f0'), suri(dst_bucket_uri, 'dst_subdir', 'existing_obj')])
            self.assertEqual(expected, actual)

    def testRecursiveCopyFilesToExistingBucketSubDirInvalidSourceParent(self):
        """Tests recursive copy of invalid paths files to existing bucket subdir."""
        src_dir1 = self.CreateTempDir(test_files=['f1'])
        src_dir2 = os.path.join(src_dir1, 'nested')
        os.mkdir(src_dir2)
        self.CreateTempFile(tmpdir=src_dir2, file_name='f2')
        dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir/existing_obj'])
        for relative_path_string in ['.', '.' + os.sep, '..', '..' + os.sep]:
            with self.subTest(relative_path_string=relative_path_string):
                invalid_parent_dir = os.path.join(src_dir2, relative_path_string)
                with self.assertRaises(InvalidUrlError):
                    self.RunCommand('cp', ['-R', src_dir1, invalid_parent_dir, suri(dst_bucket_uri, 'dst_subdir')])

    def testRecursiveCopyObjsAndFilesToNonExistentBucketSubDir(self):
        """Tests recursive copy of objs + files to non-existent bucket subdir."""
        src_bucket_uri = self.CreateBucket(test_objects=['f0', 'nested/f1'])
        src_dir = self.CreateTempDir(test_files=[('parent', 'f2'), ('parent', 'nested', 'f3')])
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', ['-R', os.path.join(src_dir, 'parent'), suri(src_bucket_uri), suri(dst_bucket_uri, 'dst_subdir')])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dst_subdir', src_bucket_uri.bucket_name, 'f0'), suri(dst_bucket_uri, 'dst_subdir', src_bucket_uri.bucket_name, 'nested', 'f1'), suri(dst_bucket_uri, 'dst_subdir', 'parent', 'f2'), suri(dst_bucket_uri, 'dst_subdir', 'parent', 'nested', 'f3')])
        self.assertEqual(expected, actual)

    def testRecursiveCopyNestedObjsToNonExistentBucketSubDir(self):
        """Tests recursive copy of objs + files to non-existent bucket subdir."""
        src_bucket_uri = self.CreateBucket(test_objects=['parent/f0', 'parent/nested/f1'])
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', ['-R', suri(src_bucket_uri, 'parent'), suri(dst_bucket_uri, 'dst_subdir')])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dst_subdir', 'f0'), suri(dst_bucket_uri, 'dst_subdir', 'nested', 'f1')])
        self.assertEqual(expected, actual)

    def testRecursiveCopyFilesToNonExistentBucketSubDir(self):
        """Tests recursive copy of objs + files to non-existent bucket subdir."""
        src_dir = self.CreateTempDir(test_files=['f2', ('nested', 'f3')])
        dst_bucket_uri = self.CreateBucket()
        self.RunCommand('cp', ['-R', src_dir, suri(dst_bucket_uri, 'dst_subdir')])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dst_subdir', 'f2'), suri(dst_bucket_uri, 'dst_subdir', 'nested', 'f3')])
        self.assertEqual(expected, actual)

    def testCopyingBucketSubDirToDir(self):
        """Tests copying a bucket subdir to a directory."""
        src_bucket_uri = self.CreateBucket(test_objects=['src_subdir/obj'])
        dst_dir = self.CreateTempDir()
        for final_src_char, final_dst_char in (('', ''), ('', '/'), ('/', ''), ('/', '/')):
            with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
                self.RunCommand('cp', ['-R', suri(src_bucket_uri, 'src_subdir') + final_src_char, dst_dir + final_dst_char])
            actual = set((str(u) for u in self._test_wildcard_iterator('%s%s**' % (dst_dir, os.sep)).IterAll(expand_top_level_buckets=True)))
            expected = set([suri(dst_dir, 'src_subdir', 'obj')])
            self.assertEqual(expected, actual)

    def testCopyingWildcardSpecifiedBucketSubDirToExistingDir(self):
        """Tests copying a wildcard-specified bucket subdir to a directory."""
        src_bucket_uri = self.CreateBucket(test_objects=['src_sub0dir/foo', 'src_sub1dir/foo', 'src_sub2dir/foo', 'src_sub3dir/foo'])
        dst_dir = self.CreateTempDir()
        for i, (final_src_char, final_dst_char) in enumerate((('', ''), ('', '/'), ('/', ''), ('/', '/'))):
            with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
                self.RunCommand('cp', ['-R', suri(src_bucket_uri, 'src_sub%d*' % i) + final_src_char, dst_dir + final_dst_char])
            actual = set((str(u) for u in self._test_wildcard_iterator(os.path.join(dst_dir, 'src_sub%ddir' % i, '**')).IterAll(expand_top_level_buckets=True)))
            expected = set([suri(dst_dir, 'src_sub%ddir' % i, 'foo')])
            self.assertEqual(expected, actual)

    def testCopyingBucketSubDirToDirFailsWithoutMinusR(self):
        """Tests for failure when attempting bucket subdir copy without -R."""
        src_bucket_uri = self.CreateBucket(test_objects=['src_subdir/obj'])
        dst_dir = self.CreateTempDir()
        try:
            self.RunCommand('cp', [suri(src_bucket_uri, 'src_subdir'), dst_dir])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertIn(NO_URLS_MATCHED_GENERIC, e.reason)

    def testCopyingBucketSubDirToBucketSubDir(self):
        """Tests copying a bucket subdir to another bucket subdir."""
        src_bucket_uri = self.CreateBucket(test_objects=['src_subdir_%d/obj' % i for i in range(4)])
        dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir_%d/obj2' % i for i in range(4)])
        for i, (final_src_char, final_dst_char) in enumerate((('', ''), ('', '/'), ('/', ''), ('/', '/'))):
            self.RunCommand('cp', ['-R', suri(src_bucket_uri, 'src_subdir_%d' % i) + final_src_char, suri(dst_bucket_uri, 'dst_subdir_%d' % i) + final_dst_char])
            actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, 'dst_subdir_%d' % i, '**')).IterAll(expand_top_level_buckets=True)))
            expected = set([suri(dst_bucket_uri, 'dst_subdir_%d' % i, 'src_subdir_%d' % i, 'obj'), suri(dst_bucket_uri, 'dst_subdir_%d' % i, 'obj2')])
            self.assertEqual(expected, actual)

    def testCopyingBucketSubDirToBucketSubDirWithNested(self):
        """Tests copying a bucket subdir to another bucket subdir with nesting."""
        src_bucket_uri = self.CreateBucket(test_objects=['src_subdir_%d/obj' % i for i in range(4)] + ['src_subdir_%d/nested/obj' % i for i in range(4)])
        dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir_%d/obj2' % i for i in range(4)])
        for i, (final_src_char, final_dst_char) in enumerate((('', ''), ('', '/'), ('/', ''), ('/', '/'))):
            self.RunCommand('cp', ['-R', suri(src_bucket_uri, 'src_subdir_%d' % i) + final_src_char, suri(dst_bucket_uri, 'dst_subdir_%d' % i) + final_dst_char])
            actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, 'dst_subdir_%d' % i, '**')).IterAll(expand_top_level_buckets=True)))
            expected = set([suri(dst_bucket_uri, 'dst_subdir_%d' % i, 'src_subdir_%d' % i, 'obj'), suri(dst_bucket_uri, 'dst_subdir_%d' % i, 'src_subdir_%d' % i, 'nested', 'obj'), suri(dst_bucket_uri, 'dst_subdir_%d' % i, 'obj2')])
            self.assertEqual(expected, actual)

    def testMovingBucketSubDirToExistingBucketSubDir(self):
        """Tests moving a bucket subdir to a existing bucket subdir."""
        src_objs = ['foo']
        for i in range(4):
            src_objs.extend(['src_subdir%d/foo2' % i, 'src_subdir%d/nested/foo3' % i])
        src_bucket_uri = self.CreateBucket(test_objects=src_objs)
        dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir%d/existing' % i for i in range(4)])
        for i, (final_src_char, final_dst_char) in enumerate((('', ''), ('', '/'), ('/', ''), ('/', '/'))):
            self.RunCommand('mv', [suri(src_bucket_uri, 'src_subdir%d' % i) + final_src_char, suri(dst_bucket_uri, 'dst_subdir%d' % i) + final_dst_char])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set()
        for i in range(4):
            expected.add(suri(dst_bucket_uri, 'dst_subdir%d' % i, 'existing'))
            expected.add(suri(dst_bucket_uri, 'dst_subdir%d' % i, 'src_subdir%d' % i, 'foo2'))
            expected.add(suri(dst_bucket_uri, 'dst_subdir%d' % i, 'src_subdir%d' % i, 'nested', 'foo3'))
        self.assertEqual(expected, actual)

    def testCopyingObjectToBucketSubDir(self):
        """Tests copying an object to a bucket subdir."""
        src_bucket_uri = self.CreateBucket(test_objects=['obj0'])
        dst_bucket_uri = self.CreateBucket(test_objects=['dir0/existing', 'dir1/existing'])
        for i, final_dst_char in enumerate(('', '/')):
            self.RunCommand('cp', [suri(src_bucket_uri, 'obj0'), suri(dst_bucket_uri, 'dir%d' % i) + final_dst_char])
            actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, 'dir%d' % i, '**')).IterAll(expand_top_level_buckets=True)))
            expected = set([suri(dst_bucket_uri, 'dir%d' % i, 'obj0'), suri(dst_bucket_uri, 'dir%d' % i, 'existing')])
            self.assertEqual(expected, actual)

    def testCopyingWildcardedFilesToBucketSubDir(self):
        """Tests copying wildcarded files to a bucket subdir."""
        dst_bucket_uri = self.CreateBucket(test_objects=['subdir0/existing', 'subdir1/existing'])
        src_dir = self.CreateTempDir(test_files=['f0', 'f1', 'f2'])
        for i, final_dst_char in enumerate(('', '/')):
            self.RunCommand('cp', [os.path.join(src_dir, 'f?'), suri(dst_bucket_uri, 'subdir%d' % i) + final_dst_char])
            actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, 'subdir%d' % i, '**')).IterAll(expand_top_level_buckets=True)))
            expected = set([suri(dst_bucket_uri, 'subdir%d' % i, 'existing'), suri(dst_bucket_uri, 'subdir%d' % i, 'f0'), suri(dst_bucket_uri, 'subdir%d' % i, 'f1'), suri(dst_bucket_uri, 'subdir%d' % i, 'f2')])
            self.assertEqual(expected, actual)

    def testCopyingOneNestedFileToBucketSubDir(self):
        """Tests copying one nested file to a bucket subdir."""
        dst_bucket_uri = self.CreateBucket(test_objects=['d0/placeholder', 'd1/placeholder'])
        src_dir = self.CreateTempDir(test_files=[('d3', 'd4', 'nested', 'f1')])
        for i, final_dst_char in enumerate(('', '/')):
            self.RunCommand('cp', ['-r', suri(src_dir, 'd3'), suri(dst_bucket_uri, 'd%d' % i) + final_dst_char])
            actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'd0', 'placeholder'), suri(dst_bucket_uri, 'd1', 'placeholder'), suri(dst_bucket_uri, 'd0', 'd3', 'd4', 'nested', 'f1'), suri(dst_bucket_uri, 'd1', 'd3', 'd4', 'nested', 'f1')])
        self.assertEqual(expected, actual)

    def testMovingWildcardedFilesToNonExistentBucketSubDir(self):
        """Tests moving files to a non-existent bucket subdir."""
        src_bucket_uri = self.CreateBucket(test_objects=['f0f0', 'f0f1', 'f1f0', 'f1f1'])
        dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir0/existing_obj', 'dst_subdir1/existing_obj'])
        for i, final_dst_char in enumerate(('', '/')):
            self.RunCommand('cp', [suri(src_bucket_uri, 'f%df*' % i), suri(dst_bucket_uri, 'dst_subdir%d' % i) + final_dst_char])
            self.RunCommand('mv', [suri(src_bucket_uri, 'f%d*' % i), suri(dst_bucket_uri, 'nonexisting%d' % i) + final_dst_char])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dst_subdir0', 'existing_obj'), suri(dst_bucket_uri, 'dst_subdir0', 'f0f0'), suri(dst_bucket_uri, 'dst_subdir0', 'f0f1'), suri(dst_bucket_uri, 'nonexisting0', 'f0f0'), suri(dst_bucket_uri, 'nonexisting0', 'f0f1'), suri(dst_bucket_uri, 'dst_subdir1', 'existing_obj'), suri(dst_bucket_uri, 'dst_subdir1', 'f1f0'), suri(dst_bucket_uri, 'dst_subdir1', 'f1f1'), suri(dst_bucket_uri, 'nonexisting1', 'f1f0'), suri(dst_bucket_uri, 'nonexisting1', 'f1f1')])
        self.assertEqual(expected, actual)

    def testMovingObjectToBucketSubDir(self):
        """Tests moving an object to a bucket subdir."""
        src_bucket_uri = self.CreateBucket(test_objects=['obj0', 'obj1'])
        dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir0/existing_obj', 'dst_subdir1/existing_obj'])
        for i, final_dst_char in enumerate(('', '/')):
            self.RunCommand('mv', [suri(src_bucket_uri, 'obj%d' % i), suri(dst_bucket_uri, 'dst_subdir%d' % i) + final_dst_char])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'dst_subdir0', 'existing_obj'), suri(dst_bucket_uri, 'dst_subdir0', 'obj0'), suri(dst_bucket_uri, 'dst_subdir1', 'existing_obj'), suri(dst_bucket_uri, 'dst_subdir1', 'obj1')])
        self.assertEqual(expected, actual)
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(src_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(actual, set())

    def testMovingBucketSubDirToNonExistentBucketSubDir(self):
        """Tests moving a bucket subdir to a non-existent bucket subdir."""
        src_bucket = self.CreateBucket(test_objects=['foo', 'src_subdir0/foo2', 'src_subdir0/nested/foo3', 'src_subdir1/foo2', 'src_subdir1/nested/foo3'])
        dst_bucket = self.CreateBucket()
        for i, final_src_char in enumerate(('', '/')):
            self.RunCommand('mv', [suri(src_bucket, 'src_subdir%d' % i) + final_src_char, suri(dst_bucket, 'dst_subdir%d' % i)])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket, 'dst_subdir0', 'foo2'), suri(dst_bucket, 'dst_subdir1', 'foo2'), suri(dst_bucket, 'dst_subdir0', 'nested', 'foo3'), suri(dst_bucket, 'dst_subdir1', 'nested', 'foo3')])
        self.assertEqual(expected, actual)

    def testRemovingBucketSubDir(self):
        """Tests removing a bucket subdir."""
        dst_bucket_uri = self.CreateBucket(test_objects=['f0', 'dir0/f1', 'dir0/nested/f2', 'dir1/f1', 'dir1/nested/f2'])
        for i, final_src_char in enumerate(('', '/')):
            self.RunCommand('rm', ['-R', suri(dst_bucket_uri, 'dir%d' % i) + final_src_char])
        actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_bucket_uri, 'f0')])
        self.assertEqual(expected, actual)

    def testRecursiveRemoveObjsInBucket(self):
        """Tests removing all objects in bucket via rm -R gs://bucket."""
        bucket_uris = [self.CreateBucket(test_objects=['f0', 'dir/f1', 'dir/nested/f2']), self.CreateBucket(test_objects=['f0', 'dir/f1', 'dir/nested/f2'])]
        for i, final_src_char in enumerate(('', '/')):
            self.RunCommand('rm', ['-R', suri(bucket_uris[i]) + final_src_char])
            try:
                self.RunCommand('ls', [suri(bucket_uris[i])])
                self.assertTrue(False)
            except NotFoundException as e:
                self.assertEqual(e.status, 404)

    def testUnicodeArgs(self):
        """Tests that you can list an object with unicode characters."""
        object_name = 'フォ'
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name=object_name, contents=b'foo')
        stdout = self.RunCommand('ls', [suri(bucket_uri, object_name)], return_stdout=True)
        self.assertIn(six.ensure_text(object_name), six.ensure_text(stdout))

    def testRecursiveListTrailingSlash(self):
        bucket_uri = self.CreateBucket()
        obj_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='/', contents=b'foo')
        stdout = self.RunCommand('ls', ['-R', suri(bucket_uri)], return_stdout=True)
        self.assertEqual(stdout.splitlines(), [suri(obj_uri) + '/:', suri(obj_uri) + '/'])

    def FinalObjNameComponent(self, uri):
        """For gs://bucket/abc/def/ghi returns ghi."""
        return uri.uri.rpartition('/')[-1]

    def testFileContainingColon(self):
        url_str = 'abc:def'
        url = StorageUrlFromString(url_str)
        self.assertEqual('file', url.scheme)
        self.assertEqual('file://%s' % url_str, url.url_string)