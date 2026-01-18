from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.commands.cp import DestinationInfo
from gslib.name_expansion import CopyObjectsIterator
from gslib.name_expansion import NameExpansionIteratorDestinationTuple
from gslib.name_expansion import NameExpansionResult
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
class TestCopyObjectsIterator(testcase.GsUtilUnitTestCase):
    """Unit tests for CopyObjectsIterator."""

    def setUp(self):
        super(TestCopyObjectsIterator, self).setUp()

    def test_iterator(self):
        src_strings_array = [['src_{}_{}'.format(i, j) for j in range(4)] for i in range(3)]
        dst_strings = ['dest_' + str(i) for i in range(3)]
        copy_objects_iterator = CopyObjectsIterator(_ConstrcutNameExpansionIteratorDestinationTupleIterator(src_strings_array, dst_strings), False)
        src_dst_strings = [(src, dst) for src_strings, dst in zip(src_strings_array, dst_strings) for src in src_strings]
        for src_string, dst_string in src_dst_strings:
            copy_object_info = next(copy_objects_iterator)
            self.assertEqual(src_string, copy_object_info.source_storage_url.object_name)
            self.assertEqual(dst_string, copy_object_info.exp_dst_url.object_name)
        iterator_ended = False
        try:
            next(copy_objects_iterator)
        except StopIteration:
            iterator_ended = True
        self.assertTrue(iterator_ended)

    def test_iterator_metadata(self):
        src_strings_array = [['gs://bucket1'], ['source'], ['s3://bucket1']]
        dst_strings = ['gs://bucket2', 'dest', 'gs://bucket2']
        copy_objects_iterator = CopyObjectsIterator(_ConstrcutNameExpansionIteratorDestinationTupleIterator(src_strings_array, dst_strings), False)
        self.assertFalse(copy_objects_iterator.has_cloud_src)
        self.assertFalse(copy_objects_iterator.has_file_src)
        self.assertEqual(len(copy_objects_iterator.provider_types), 0)
        next(copy_objects_iterator)
        self.assertTrue(copy_objects_iterator.has_cloud_src)
        self.assertFalse(copy_objects_iterator.has_file_src)
        self.assertEqual(len(copy_objects_iterator.provider_types), 1)
        self.assertTrue('gs' in copy_objects_iterator.provider_types)
        next(copy_objects_iterator)
        self.assertTrue(copy_objects_iterator.has_cloud_src)
        self.assertTrue(copy_objects_iterator.has_file_src)
        self.assertEqual(len(copy_objects_iterator.provider_types), 2)
        self.assertTrue('file' in copy_objects_iterator.provider_types)
        self.assertFalse(copy_objects_iterator.is_daisy_chain)
        next(copy_objects_iterator)
        self.assertEqual(len(copy_objects_iterator.provider_types), 3)
        self.assertTrue('s3' in copy_objects_iterator.provider_types)
        self.assertTrue(copy_objects_iterator.is_daisy_chain)