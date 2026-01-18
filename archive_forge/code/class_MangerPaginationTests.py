import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
class MangerPaginationTests(ManagerTest):

    def setUp(self):
        super(MangerPaginationTests, self).setUp()
        self.manager = base.Manager()
        self.manager.api = mock.Mock()
        self.manager.api.client = mock.Mock()
        self.manager.resource_class = base.Resource
        self.response_key = 'response_key'
        self.data = [{'foo': 'p1'}, {'foo': 'p2'}]
        self.next_data = [{'foo': 'p3'}, {'foo': 'p4'}]
        self.marker = 'test-marker'
        self.limit = '20'
        self.url = 'http://test_url'
        self.next_url = '%s?marker=%s&limit=%s' % (self.url, self.marker, self.limit)
        self.links = [{'href': self.next_url, 'rel': 'next'}]
        self.body = {self.response_key: self.data, 'links': self.links}
        self.next_body = {self.response_key: self.next_data}

        def side_effect(url):
            if url == self.url:
                return (None, self.body)
            if 'marker=%s' % self.marker in url and 'limit=%s' % self.limit in url:
                self.next_url = url
                return (None, self.next_body)
        self.manager.api.client.get = mock.Mock(side_effect=side_effect)

    def tearDown(self):
        super(MangerPaginationTests, self).tearDown()

    def test_pagination(self):
        resp = self.manager._paginated(self.url, self.response_key)
        self.manager.api.client.get.assert_called_with(self.url)
        self.assertEqual('p1', resp[0].foo)
        self.assertEqual('p2', resp[1].foo)
        self.assertEqual(self.marker, resp.next)
        self.assertEqual(self.links, resp.links)
        self.assertIsInstance(resp, common.Paginated)

    def test_pagination_next(self):
        resp = self.manager._paginated(self.url, self.response_key, limit=self.limit, marker=self.marker)
        self.manager.api.client.get.assert_called_with(self.next_url)
        self.assertEqual('p3', resp[0].foo)
        self.assertEqual('p4', resp[1].foo)
        self.assertIsNone(resp.next)
        self.assertEqual([], resp.links)
        self.assertIsInstance(resp, common.Paginated)

    def test_pagination_error(self):
        self.manager.api.client.get = mock.Mock(return_value=(None, None))
        self.assertRaises(Exception, self.manager._paginated, self.url, self.response_key)