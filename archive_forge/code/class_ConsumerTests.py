from unittest import mock
import fixtures
from urllib import parse as urlparse
import uuid
from testtools import matchers
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient import utils as client_utils
from keystoneclient.v3.contrib.oauth1 import access_tokens
from keystoneclient.v3.contrib.oauth1 import auth
from keystoneclient.v3.contrib.oauth1 import consumers
from keystoneclient.v3.contrib.oauth1 import request_tokens
class ConsumerTests(utils.ClientTestCase, utils.CrudTests):

    def setUp(self):
        if oauth1 is None:
            self.skipTest('oauthlib package not available')
        super(ConsumerTests, self).setUp()
        self.key = 'consumer'
        self.collection_key = 'consumers'
        self.model = consumers.Consumer
        self.manager = self.client.oauth1.consumers
        self.path_prefix = 'OS-OAUTH1'

    def new_ref(self, **kwargs):
        kwargs = super(ConsumerTests, self).new_ref(**kwargs)
        kwargs.setdefault('description', uuid.uuid4().hex)
        return kwargs

    def test_description_is_optional(self):
        consumer_id = uuid.uuid4().hex
        resp_ref = {'consumer': {'description': None, 'id': consumer_id}}
        self.stub_url('POST', [self.path_prefix, self.collection_key], status_code=201, json=resp_ref)
        consumer = self.manager.create()
        self.assertEqual(consumer_id, consumer.id)
        self.assertIsNone(consumer.description)

    def test_description_not_included(self):
        consumer_id = uuid.uuid4().hex
        resp_ref = {'consumer': {'id': consumer_id}}
        self.stub_url('POST', [self.path_prefix, self.collection_key], status_code=201, json=resp_ref)
        consumer = self.manager.create()
        self.assertEqual(consumer_id, consumer.id)