import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
class PaginatedTest(testtools.TestCase):

    def setUp(self):
        super(PaginatedTest, self).setUp()
        self.items_ = ['item1', 'item2']
        self.next_marker_ = 'next-marker'
        self.links_ = ['link1', 'link2']
        self.pgn = common.Paginated(self.items_, self.next_marker_, self.links_)

    def tearDown(self):
        super(PaginatedTest, self).tearDown()

    def test___init__(self):
        self.assertEqual(self.items_, self.pgn.items)
        self.assertEqual(self.next_marker_, self.pgn.next)
        self.assertEqual(self.links_, self.pgn.links)

    def test___len__(self):
        self.assertEqual(len(self.items_), self.pgn.__len__())

    def test___iter__(self):
        itr_expected = self.items_.__iter__()
        itr = self.pgn.__iter__()
        self.assertEqual(next(itr_expected), next(itr))
        self.assertEqual(next(itr_expected), next(itr))
        self.assertRaises(StopIteration, next, itr_expected)
        self.assertRaises(StopIteration, next, itr)

    def test___getitem__(self):
        self.assertEqual(self.items_[0], self.pgn.__getitem__(0))

    def test___setitem__(self):
        self.pgn.__setitem__(0, 'new-item')
        self.assertEqual('new-item', self.pgn.items[0])

    def test___delitem(self):
        del self.pgn[0]
        self.assertEqual(1, self.pgn.__len__())

    def test___reversed__(self):
        itr = self.pgn.__reversed__()
        self.assertEqual('item2', next(itr))
        self.assertEqual('item1', next(itr))
        self.assertRaises(StopIteration, next, itr)

    def test___contains__(self):
        self.assertTrue(self.pgn.__contains__('item1'))
        self.assertTrue(self.pgn.__contains__('item2'))
        self.assertFalse(self.pgn.__contains__('item3'))