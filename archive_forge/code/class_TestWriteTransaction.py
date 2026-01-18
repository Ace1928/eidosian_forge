import breezy.errors as errors
import breezy.transactions as transactions
from breezy.tests import TestCase
class TestWriteTransaction(TestCase):

    def setUp(self):
        self.transaction = transactions.WriteTransaction()
        super().setUp()

    def test_register_clean(self):
        self.transaction.register_clean('anobject')

    def test_register_dirty(self):
        self.transaction.register_dirty('anobject')

    def test_map(self):
        self.assertNotEqual(None, getattr(self.transaction, 'map', None))

    def test_add_and_get(self):
        weave = 'a weave'
        self.transaction.map.add_weave('id', weave)
        self.assertEqual(weave, self.transaction.map.find_weave('id'))

    def test_finish_returns(self):
        self.transaction.finish()

    def test_finish_tells_versioned_file_finished(self):
        weave = DummyWeave('a weave')
        self.transaction.register_dirty(weave)
        self.transaction.finish()
        self.assertTrue(weave.finished)

    def test_zero_size_cache(self):
        self.transaction.set_cache_size(0)
        weave = DummyWeave('a weave')
        self.transaction.map.add_weave('id', weave)
        self.assertEqual(weave, self.transaction.map.find_weave('id'))
        weave = None
        self.transaction.register_clean(self.transaction.map.find_weave('id'))
        self.assertEqual(None, self.transaction.map.find_weave('id'))
        weave = DummyWeave('another weave')
        self.transaction.map.add_weave('id', weave)
        self.transaction.register_clean(self.transaction.map.find_weave('id'))
        self.assertEqual(weave, self.transaction.map.find_weave('id'))
        del weave
        self.assertEqual(DummyWeave('another weave'), self.transaction.map.find_weave('id'))

    def test_zero_size_cache_dirty_objects(self):
        self.transaction.set_cache_size(0)
        weave = DummyWeave('a weave')
        self.transaction.map.add_weave('id', weave)
        self.assertEqual(weave, self.transaction.map.find_weave('id'))
        weave = None
        self.transaction.register_dirty(self.transaction.map.find_weave('id'))
        self.assertNotEqual(None, self.transaction.map.find_weave('id'))

    def test_clean_to_dirty(self):
        weave = DummyWeave('A weave')
        self.transaction.map.add_weave('id', weave)
        self.transaction.register_clean(weave)
        self.transaction.register_dirty(weave)
        self.assertTrue(self.transaction.is_dirty(weave))
        self.assertFalse(self.transaction.is_clean(weave))

    def test_small_cache(self):
        self.transaction.set_cache_size(1)
        self.transaction.map.add_weave('id', DummyWeave('a weave'))
        self.transaction.register_clean(self.transaction.map.find_weave('id'))
        self.assertEqual(DummyWeave('a weave'), self.transaction.map.find_weave('id'))
        self.transaction.map.add_weave('id2', DummyWeave('a weave also'))
        self.transaction.register_clean(self.transaction.map.find_weave('id2'))
        self.assertEqual(None, self.transaction.map.find_weave('id'))
        self.assertEqual(DummyWeave('a weave also'), self.transaction.map.find_weave('id2'))

    def test_small_cache_with_references(self):
        weave = 'a weave'
        weave2 = 'another weave'
        self.transaction.map.add_weave('id', weave)
        self.transaction.map.add_weave('id2', weave2)
        self.assertEqual(weave, self.transaction.map.find_weave('id'))
        self.assertEqual(weave2, self.transaction.map.find_weave('id2'))
        weave = None
        self.assertEqual('a weave', self.transaction.map.find_weave('id'))

    def test_precious_with_zero_size_cache(self):
        self.transaction.set_cache_size(0)
        weave = DummyWeave('a weave')
        self.transaction.map.add_weave('id', weave)
        self.assertEqual(weave, self.transaction.map.find_weave('id'))
        weave = None
        self.transaction.register_clean(self.transaction.map.find_weave('id'), precious=True)
        self.assertEqual(DummyWeave('a weave'), self.transaction.map.find_weave('id'))

    def test_writable(self):
        self.assertTrue(self.transaction.writeable())