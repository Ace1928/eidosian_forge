import logging
import sys
import urllib.parse as urlparse
import uuid
import fixtures
from oslo_serialization import jsonutils
import requests
import requests_mock
from requests_mock.contrib import fixture
import testscenarios
import testtools
from keystoneclient.tests.unit import client_fixtures
class ClientTestCaseMixin(testscenarios.WithScenarios):
    client_fixture_class = None
    data_fixture_class = None

    def setUp(self):
        super(ClientTestCaseMixin, self).setUp()
        self.data_fixture = None
        self.client_fixture = None
        self.client = None
        if self.client_fixture_class:
            fix = self.client_fixture_class(self.requests_mock, self.deprecations)
            self.client_fixture = self.useFixture(fix)
            self.client = self.client_fixture.client
            self.TEST_USER_ID = self.client_fixture.user_id
        if self.data_fixture_class:
            fix = self.data_fixture_class(self.requests_mock)
            self.data_fixture = self.useFixture(fix)