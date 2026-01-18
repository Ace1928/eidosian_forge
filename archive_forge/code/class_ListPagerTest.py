import unittest
from apitools.base.py import list_pager
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 \
from samples.fusiontables_sample.fusiontables_v1 \
from samples.iam_sample.iam_v1 import iam_v1_client as iam_client
from samples.iam_sample.iam_v1 import iam_v1_messages as iam_messages
class ListPagerTest(unittest.TestCase):

    def _AssertInstanceSequence(self, results, n):
        counter = 0
        for instance in results:
            self.assertEqual(instance.name, 'c' + str(counter))
            counter += 1
        self.assertEqual(counter, n)

    def setUp(self):
        self.mocked_client = mock.Client(fusiontables.FusiontablesV1)
        self.mocked_client.Mock()
        self.addCleanup(self.mocked_client.Unmock)

    def testYieldFromList(self):
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=100, pageToken=None, tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c0'), messages.Column(name='c1'), messages.Column(name='c2'), messages.Column(name='c3')], nextPageToken='x'))
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=100, pageToken='x', tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c4'), messages.Column(name='c5'), messages.Column(name='c6'), messages.Column(name='c7')]))
        client = fusiontables.FusiontablesV1(get_credentials=False)
        request = messages.FusiontablesColumnListRequest(tableId='mytable')
        results = list_pager.YieldFromList(client.column, request)
        self._AssertInstanceSequence(results, 8)

    def testYieldNoRecords(self):
        client = fusiontables.FusiontablesV1(get_credentials=False)
        request = messages.FusiontablesColumnListRequest(tableId='mytable')
        results = list_pager.YieldFromList(client.column, request, limit=False)
        self.assertEqual(0, len(list(results)))

    def testYieldFromListPartial(self):
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=6, pageToken=None, tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c0'), messages.Column(name='c1'), messages.Column(name='c2'), messages.Column(name='c3')], nextPageToken='x'))
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=2, pageToken='x', tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c4'), messages.Column(name='c5'), messages.Column(name='c6'), messages.Column(name='c7')]))
        client = fusiontables.FusiontablesV1(get_credentials=False)
        request = messages.FusiontablesColumnListRequest(tableId='mytable')
        results = list_pager.YieldFromList(client.column, request, limit=6)
        self._AssertInstanceSequence(results, 6)

    def testYieldFromListPaging(self):
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=5, pageToken=None, tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c0'), messages.Column(name='c1'), messages.Column(name='c2'), messages.Column(name='c3'), messages.Column(name='c4')], nextPageToken='x'))
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=4, pageToken='x', tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c5'), messages.Column(name='c6'), messages.Column(name='c7'), messages.Column(name='c8')]))
        client = fusiontables.FusiontablesV1(get_credentials=False)
        request = messages.FusiontablesColumnListRequest(tableId='mytable')
        results = list_pager.YieldFromList(client.column, request, limit=9, batch_size=5)
        self._AssertInstanceSequence(results, 9)

    def testYieldFromListBatchSizeNone(self):
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=None, pageToken=None, tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c0'), messages.Column(name='c1'), messages.Column(name='c2'), messages.Column(name='c3'), messages.Column(name='c4'), messages.Column(name='c5'), messages.Column(name='c6')], nextPageToken='x'))
        client = fusiontables.FusiontablesV1(get_credentials=False)
        request = messages.FusiontablesColumnListRequest(tableId='mytable')
        results = list_pager.YieldFromList(client.column, request, limit=5, batch_size=None)
        self._AssertInstanceSequence(results, 5)

    def testYieldFromListEmpty(self):
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=6, pageToken=None, tableId='mytable'), messages.ColumnList())
        client = fusiontables.FusiontablesV1(get_credentials=False)
        request = messages.FusiontablesColumnListRequest(tableId='mytable')
        results = list_pager.YieldFromList(client.column, request, limit=6)
        self._AssertInstanceSequence(results, 0)

    def testYieldFromListWithPredicate(self):
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=100, pageToken=None, tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c0'), messages.Column(name='bad0'), messages.Column(name='c1'), messages.Column(name='bad1')], nextPageToken='x'))
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=100, pageToken='x', tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c2')]))
        client = fusiontables.FusiontablesV1(get_credentials=False)
        request = messages.FusiontablesColumnListRequest(tableId='mytable')
        results = list_pager.YieldFromList(client.column, request, predicate=lambda x: 'c' in x.name)
        self._AssertInstanceSequence(results, 3)

    def testYieldFromListWithCustomGetFieldFunction(self):
        self.mocked_client.column.List.Expect(messages.FusiontablesColumnListRequest(maxResults=100, pageToken=None, tableId='mytable'), messages.ColumnList(items=[messages.Column(name='c0')]))
        custom_getter_called = []

        def Custom_Getter(message, attribute):
            custom_getter_called.append(True)
            return getattr(message, attribute)
        client = fusiontables.FusiontablesV1(get_credentials=False)
        request = messages.FusiontablesColumnListRequest(tableId='mytable')
        results = list_pager.YieldFromList(client.column, request, get_field_func=Custom_Getter)
        self._AssertInstanceSequence(results, 1)
        self.assertEquals(1, len(custom_getter_called))