import copy
import os
import random
import re
import subprocess
from testtools import matchers
from unittest import mock
import uuid
import fixtures
import flask
import http.client
from lxml import etree
from oslo_serialization import jsonutils
from oslo_utils import importutils
import saml2
from saml2 import saml
from saml2 import sigver
import urllib
from keystone.api._shared import authentication
from keystone.api import auth as auth_api
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import render_token
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.models import token_model
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import core
from keystone.tests.unit import federation_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
class ServiceProviderTests(test_v3.RestfulTestCase):
    """A test class for Service Providers."""
    MEMBER_NAME = 'service_provider'
    COLLECTION_NAME = 'service_providers'
    SERVICE_PROVIDER_ID = 'ACME'
    SP_KEYS = ['auth_url', 'id', 'enabled', 'description', 'relay_state_prefix', 'sp_url']

    def setUp(self):
        super(ServiceProviderTests, self).setUp()
        url = self.base_url(suffix=self.SERVICE_PROVIDER_ID)
        self.SP_REF = core.new_service_provider_ref()
        self.SERVICE_PROVIDER = self.put(url, body={'service_provider': self.SP_REF}, expected_status=http.client.CREATED).result

    def base_url(self, suffix=None):
        if suffix is not None:
            return '/OS-FEDERATION/service_providers/' + str(suffix)
        return '/OS-FEDERATION/service_providers'

    def _create_default_sp(self, body=None):
        """Create default Service Provider."""
        url = self.base_url(suffix=uuid.uuid4().hex)
        if body is None:
            body = core.new_service_provider_ref()
        resp = self.put(url, body={'service_provider': body}, expected_status=http.client.CREATED)
        return resp

    def test_get_head_service_provider(self):
        url = self.base_url(suffix=self.SERVICE_PROVIDER_ID)
        resp = self.get(url)
        self.assertValidEntity(resp.result['service_provider'], keys_to_check=self.SP_KEYS)
        resp = self.head(url, expected_status=http.client.OK)

    def test_get_service_provider_fail(self):
        url = self.base_url(suffix=uuid.uuid4().hex)
        self.get(url, expected_status=http.client.NOT_FOUND)

    def test_create_service_provider(self):
        url = self.base_url(suffix=uuid.uuid4().hex)
        sp = core.new_service_provider_ref()
        resp = self.put(url, body={'service_provider': sp}, expected_status=http.client.CREATED)
        self.assertValidEntity(resp.result['service_provider'], keys_to_check=self.SP_KEYS)

    @unit.skip_if_cache_disabled('federation')
    def test_create_service_provider_invalidates_cache(self):
        resp = self.get(self.base_url(), expected_status=http.client.OK)
        self.assertThat(resp.json_body['service_providers'], matchers.HasLength(1))
        url = self.base_url(suffix=uuid.uuid4().hex)
        sp = core.new_service_provider_ref()
        self.put(url, body={'service_provider': sp}, expected_status=http.client.CREATED)
        resp = self.get(self.base_url(), expected_status=http.client.OK)
        self.assertThat(resp.json_body['service_providers'], matchers.HasLength(2))

    @unit.skip_if_cache_disabled('federation')
    def test_delete_service_provider_invalidates_cache(self):
        resp = self.get(self.base_url(), expected_status=http.client.OK)
        self.assertThat(resp.json_body['service_providers'], matchers.HasLength(1))
        url = self.base_url(suffix=uuid.uuid4().hex)
        sp = core.new_service_provider_ref()
        self.put(url, body={'service_provider': sp}, expected_status=http.client.CREATED)
        resp = self.get(self.base_url(), expected_status=http.client.OK)
        self.assertThat(resp.json_body['service_providers'], matchers.HasLength(2))
        self.delete(url, expected_status=http.client.NO_CONTENT)
        resp = self.get(self.base_url(), expected_status=http.client.OK)
        self.assertThat(resp.json_body['service_providers'], matchers.HasLength(1))

    @unit.skip_if_cache_disabled('federation')
    def test_update_service_provider_invalidates_cache(self):
        resp = self.get(self.base_url(), expected_status=http.client.OK)
        self.assertThat(resp.json_body['service_providers'], matchers.HasLength(1))
        service_provider_id = uuid.uuid4().hex
        url = self.base_url(suffix=service_provider_id)
        sp = core.new_service_provider_ref()
        self.put(url, body={'service_provider': sp}, expected_status=http.client.CREATED)
        resp = self.get(self.base_url(), expected_status=http.client.OK)
        self.assertThat(resp.json_body['service_providers'], matchers.HasLength(2))
        updated_description = uuid.uuid4().hex
        body = {'service_provider': {'description': updated_description}}
        self.patch(url, body=body, expected_status=http.client.OK)
        resp = self.get(self.base_url(), expected_status=http.client.OK)
        self.assertThat(resp.json_body['service_providers'], matchers.HasLength(2))
        for sp in resp.json_body['service_providers']:
            if sp['id'] == service_provider_id:
                self.assertEqual(sp['description'], updated_description)

    def test_create_sp_relay_state_default(self):
        """Create an SP without relay state, should default to `ss:mem`."""
        url = self.base_url(suffix=uuid.uuid4().hex)
        sp = core.new_service_provider_ref()
        del sp['relay_state_prefix']
        resp = self.put(url, body={'service_provider': sp}, expected_status=http.client.CREATED)
        sp_result = resp.result['service_provider']
        self.assertEqual(CONF.saml.relay_state_prefix, sp_result['relay_state_prefix'])

    def test_create_sp_relay_state_non_default(self):
        """Create an SP with custom relay state."""
        url = self.base_url(suffix=uuid.uuid4().hex)
        sp = core.new_service_provider_ref()
        non_default_prefix = uuid.uuid4().hex
        sp['relay_state_prefix'] = non_default_prefix
        resp = self.put(url, body={'service_provider': sp}, expected_status=http.client.CREATED)
        sp_result = resp.result['service_provider']
        self.assertEqual(non_default_prefix, sp_result['relay_state_prefix'])

    def test_create_service_provider_fail(self):
        """Try adding SP object with unallowed attribute."""
        url = self.base_url(suffix=uuid.uuid4().hex)
        sp = core.new_service_provider_ref()
        sp[uuid.uuid4().hex] = uuid.uuid4().hex
        self.put(url, body={'service_provider': sp}, expected_status=http.client.BAD_REQUEST)

    def test_list_head_service_providers(self):
        """Test listing of service provider objects.

        Add two new service providers. List all available service providers.
        Expect to get list of three service providers (one created by setUp())
        Test if attributes match.

        """
        ref_service_providers = {uuid.uuid4().hex: core.new_service_provider_ref(), uuid.uuid4().hex: core.new_service_provider_ref()}
        for id, sp in ref_service_providers.items():
            url = self.base_url(suffix=id)
            self.put(url, body={'service_provider': sp}, expected_status=http.client.CREATED)
        ref_service_providers[self.SERVICE_PROVIDER_ID] = self.SP_REF
        for id, sp in ref_service_providers.items():
            sp['id'] = id
        url = self.base_url()
        resp = self.get(url)
        service_providers = resp.result
        for service_provider in service_providers['service_providers']:
            id = service_provider['id']
            self.assertValidEntity(service_provider, ref=ref_service_providers[id], keys_to_check=self.SP_KEYS)
        self.head(url, expected_status=http.client.OK)

    def test_update_service_provider(self):
        """Update existing service provider.

        Update default existing service provider and make sure it has been
        properly changed.

        """
        new_sp_ref = core.new_service_provider_ref()
        url = self.base_url(suffix=self.SERVICE_PROVIDER_ID)
        resp = self.patch(url, body={'service_provider': new_sp_ref})
        patch_result = resp.result
        new_sp_ref['id'] = self.SERVICE_PROVIDER_ID
        self.assertValidEntity(patch_result['service_provider'], ref=new_sp_ref, keys_to_check=self.SP_KEYS)
        resp = self.get(url)
        get_result = resp.result
        self.assertDictEqual(patch_result['service_provider'], get_result['service_provider'])

    def test_update_service_provider_immutable_parameters(self):
        """Update immutable attributes in service provider.

        In this particular case the test will try to change ``id`` attribute.
        The server should return an HTTP 403 Forbidden error code.

        """
        new_sp_ref = {'id': uuid.uuid4().hex}
        url = self.base_url(suffix=self.SERVICE_PROVIDER_ID)
        self.patch(url, body={'service_provider': new_sp_ref}, expected_status=http.client.BAD_REQUEST)

    def test_update_service_provider_unknown_parameter(self):
        new_sp_ref = core.new_service_provider_ref()
        new_sp_ref[uuid.uuid4().hex] = uuid.uuid4().hex
        url = self.base_url(suffix=self.SERVICE_PROVIDER_ID)
        self.patch(url, body={'service_provider': new_sp_ref}, expected_status=http.client.BAD_REQUEST)

    def test_update_service_provider_returns_not_found(self):
        new_sp_ref = core.new_service_provider_ref()
        new_sp_ref['description'] = uuid.uuid4().hex
        url = self.base_url(suffix=uuid.uuid4().hex)
        self.patch(url, body={'service_provider': new_sp_ref}, expected_status=http.client.NOT_FOUND)

    def test_update_sp_relay_state(self):
        """Update an SP with custom relay state."""
        new_sp_ref = core.new_service_provider_ref()
        non_default_prefix = uuid.uuid4().hex
        new_sp_ref['relay_state_prefix'] = non_default_prefix
        url = self.base_url(suffix=self.SERVICE_PROVIDER_ID)
        resp = self.patch(url, body={'service_provider': new_sp_ref})
        sp_result = resp.result['service_provider']
        self.assertEqual(non_default_prefix, sp_result['relay_state_prefix'])

    def test_delete_service_provider(self):
        url = self.base_url(suffix=self.SERVICE_PROVIDER_ID)
        self.delete(url)

    def test_delete_service_provider_returns_not_found(self):
        url = self.base_url(suffix=uuid.uuid4().hex)
        self.delete(url, expected_status=http.client.NOT_FOUND)

    def test_filter_list_sp_by_id(self):

        def get_id(resp):
            sp = resp.result.get('service_provider')
            return sp.get('id')
        sp1_id = get_id(self._create_default_sp())
        sp2_id = get_id(self._create_default_sp())
        url = self.base_url()
        resp = self.get(url)
        sps = resp.result.get('service_providers')
        entities_ids = [e['id'] for e in sps]
        self.assertIn(sp1_id, entities_ids)
        self.assertIn(sp2_id, entities_ids)
        url = self.base_url() + '?id=' + sp1_id
        resp = self.get(url)
        sps = resp.result.get('service_providers')
        entities_ids = [e['id'] for e in sps]
        self.assertIn(sp1_id, entities_ids)
        self.assertNotIn(sp2_id, entities_ids)

    def test_filter_list_sp_by_enabled(self):

        def get_id(resp):
            sp = resp.result.get('service_provider')
            return sp.get('id')
        sp1_id = get_id(self._create_default_sp())
        sp2_ref = core.new_service_provider_ref()
        sp2_ref['enabled'] = False
        sp2_id = get_id(self._create_default_sp(body=sp2_ref))
        url = self.base_url()
        resp = self.get(url)
        sps = resp.result.get('service_providers')
        entities_ids = [e['id'] for e in sps]
        self.assertIn(sp1_id, entities_ids)
        self.assertIn(sp2_id, entities_ids)
        url = self.base_url() + '?enabled=True'
        resp = self.get(url)
        sps = resp.result.get('service_providers')
        entities_ids = [e['id'] for e in sps]
        self.assertIn(sp1_id, entities_ids)
        self.assertNotIn(sp2_id, entities_ids)