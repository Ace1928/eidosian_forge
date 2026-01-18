import json
import os
from unittest import mock
import fixtures
import requests
from requests_mock.contrib import fixture as requests_mock_fixture
import testtools
class FixturedTestCase(TestCase):
    client_fixture_class = None
    data_fixture_class = None

    def setUp(self):
        super(FixturedTestCase, self).setUp()
        self.requests = self.useFixture(requests_mock_fixture.Fixture())
        self.data_fixture = None
        self.client_fixture = None
        self.cs = None
        if self.client_fixture_class:
            fix = self.client_fixture_class(self.requests)
            self.client_fixture = self.useFixture(fix)
            self.cs = self.client_fixture.new_client()
        if self.data_fixture_class:
            fix = self.data_fixture_class(self.requests)
            self.data_fixture = self.useFixture(fix)

    def assert_called(self, method, path, body=None):
        self.assertEqual(method, self.requests.last_request.method)
        self.assertEqual(path, self.requests.last_request.path_url)
        if body:
            req_data = self.requests.last_request.body
            if isinstance(req_data, bytes):
                req_data = req_data.decode('utf-8')
            if not isinstance(body, str):
                req_data = json.loads(req_data)
            self.assertEqual(body, req_data)