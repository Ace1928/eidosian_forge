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
class FederatedIdentityProviderTests(test_v3.RestfulTestCase):
    """A test class for Identity Providers."""
    idp_keys = ['description', 'enabled']
    default_body = {'description': None, 'enabled': True}

    def base_url(self, suffix=None):
        if suffix is not None:
            return '/OS-FEDERATION/identity_providers/' + str(suffix)
        return '/OS-FEDERATION/identity_providers'

    def _fetch_attribute_from_response(self, resp, parameter, assert_is_not_none=True):
        """Fetch single attribute from TestResponse object."""
        result = resp.result.get(parameter)
        if assert_is_not_none:
            self.assertIsNotNone(result)
        return result

    def _create_and_decapsulate_response(self, body=None):
        """Create IdP and fetch it's random id along with entity."""
        default_resp = self._create_default_idp(body=body)
        idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        self.assertIsNotNone(idp)
        idp_id = idp.get('id')
        return (idp_id, idp)

    def _get_idp(self, idp_id):
        """Fetch IdP entity based on its id."""
        url = self.base_url(suffix=idp_id)
        resp = self.get(url)
        return resp

    def _create_default_idp(self, body=None, expected_status=http.client.CREATED):
        """Create default IdP."""
        url = self.base_url(suffix=uuid.uuid4().hex)
        if body is None:
            body = self._http_idp_input()
        resp = self.put(url, body={'identity_provider': body}, expected_status=expected_status)
        return resp

    def _http_idp_input(self):
        """Create default input dictionary for IdP data."""
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        return body

    def _assign_protocol_to_idp(self, idp_id=None, proto=None, url=None, mapping_id=None, validate=True, **kwargs):
        if url is None:
            url = self.base_url(suffix='%(idp_id)s/protocols/%(protocol_id)s')
        if idp_id is None:
            idp_id, _ = self._create_and_decapsulate_response()
        if proto is None:
            proto = uuid.uuid4().hex
        if mapping_id is None:
            mapping_id = uuid.uuid4().hex
        self._create_mapping(mapping_id)
        body = {'mapping_id': mapping_id}
        url = url % {'idp_id': idp_id, 'protocol_id': proto}
        resp = self.put(url, body={'protocol': body}, **kwargs)
        if validate:
            self.assertValidResponse(resp, 'protocol', dummy_validator, keys_to_check=['id', 'mapping_id'], ref={'id': proto, 'mapping_id': mapping_id})
        return (resp, idp_id, proto)

    def _get_protocol(self, idp_id, protocol_id):
        url = '%s/protocols/%s' % (idp_id, protocol_id)
        url = self.base_url(suffix=url)
        r = self.get(url)
        return r

    def _create_mapping(self, mapping_id):
        mapping = mapping_fixtures.MAPPING_EPHEMERAL_USER
        mapping['id'] = mapping_id
        url = '/OS-FEDERATION/mappings/%s' % mapping_id
        self.put(url, body={'mapping': mapping}, expected_status=http.client.CREATED)

    def assertIdpDomainCreated(self, idp_id, domain_id):
        domain = PROVIDERS.resource_api.get_domain(domain_id)
        self.assertEqual(domain_id, domain['name'])
        self.assertIn(idp_id, domain['description'])

    def test_create_idp_without_domain_id(self):
        """Create the IdentityProvider entity associated to remote_ids."""
        keys_to_check = list(self.idp_keys)
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        resp = self._create_default_idp(body=body)
        self.assertValidResponse(resp, 'identity_provider', dummy_validator, keys_to_check=keys_to_check, ref=body)
        attr = self._fetch_attribute_from_response(resp, 'identity_provider')
        self.assertIdpDomainCreated(attr['id'], attr['domain_id'])

    def test_create_idp_with_domain_id(self):
        keys_to_check = list(self.idp_keys)
        keys_to_check.append('domain_id')
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        body['domain_id'] = domain['id']
        resp = self._create_default_idp(body=body)
        self.assertValidResponse(resp, 'identity_provider', dummy_validator, keys_to_check=keys_to_check, ref=body)

    def test_create_idp_domain_id_none(self):
        keys_to_check = list(self.idp_keys)
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        body['domain_id'] = None
        resp = self._create_default_idp(body=body)
        self.assertValidResponse(resp, 'identity_provider', dummy_validator, keys_to_check=keys_to_check, ref=body)
        attr = self._fetch_attribute_from_response(resp, 'identity_provider')
        self.assertIdpDomainCreated(attr['id'], attr['domain_id'])

    def test_conflicting_idp_cleans_up_auto_generated_domain(self):
        resp = self._create_default_idp()
        idp_id = resp.json_body['identity_provider']['id']
        domains = PROVIDERS.resource_api.list_domains()
        number_of_domains = len(domains)
        resp = self.put(self.base_url(suffix=idp_id), body={'identity_provider': self.default_body.copy()}, expected_status=http.client.CONFLICT)
        domains = PROVIDERS.resource_api.list_domains()
        self.assertEqual(number_of_domains, len(domains))

    def test_conflicting_idp_does_not_delete_existing_domain(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        body['domain_id'] = domain['id']
        resp = self._create_default_idp(body=body)
        idp = resp.json_body['identity_provider']
        idp_id = idp['id']
        self.assertEqual(idp['domain_id'], domain['id'])
        body = self.default_body.copy()
        body['domain_id'] = domain['id']
        resp = self.put(self.base_url(suffix=idp_id), body={'identity_provider': body}, expected_status=http.client.CONFLICT)
        self.assertIsNotNone(PROVIDERS.resource_api.get_domain(domain['id']))

    def test_create_multi_idp_to_one_domain(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        keys_to_check = list(self.idp_keys)
        keys_to_check.append('domain_id')
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        body['domain_id'] = domain['id']
        idp1 = self._create_default_idp(body=body)
        self.assertValidResponse(idp1, 'identity_provider', dummy_validator, keys_to_check=keys_to_check, ref=body)
        url = self.base_url(suffix=uuid.uuid4().hex)
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        body['domain_id'] = domain['id']
        idp2 = self.put(url, body={'identity_provider': body}, expected_status=http.client.CREATED)
        self.assertValidResponse(idp2, 'identity_provider', dummy_validator, keys_to_check=keys_to_check, ref=body)
        self.assertEqual(idp1.result['identity_provider']['domain_id'], idp2.result['identity_provider']['domain_id'])

    def test_cannot_update_idp_domain(self):
        body = self.default_body.copy()
        default_resp = self._create_default_idp(body=body)
        default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        idp_id = default_idp.get('id')
        self.assertIsNotNone(idp_id)
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        body['domain_id'] = domain['id']
        body = {'identity_provider': body}
        url = self.base_url(suffix=idp_id)
        self.patch(url, body=body, expected_status=http.client.BAD_REQUEST)

    def test_create_idp_with_nonexistent_domain_id_fails(self):
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        body['domain_id'] = uuid.uuid4().hex
        self._create_default_idp(body=body, expected_status=http.client.NOT_FOUND)

    def test_create_idp_remote(self):
        """Create the IdentityProvider entity associated to remote_ids."""
        keys_to_check = list(self.idp_keys)
        keys_to_check.append('remote_ids')
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        body['remote_ids'] = [uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex]
        resp = self._create_default_idp(body=body)
        self.assertValidResponse(resp, 'identity_provider', dummy_validator, keys_to_check=keys_to_check, ref=body)
        attr = self._fetch_attribute_from_response(resp, 'identity_provider')
        self.assertIdpDomainCreated(attr['id'], attr['domain_id'])

    def test_create_idp_remote_repeated(self):
        """Create two IdentityProvider entities with some remote_ids.

        A remote_id is the same for both so the second IdP is not
        created because of the uniqueness of the remote_ids

        Expect HTTP 409 Conflict code for the latter call.

        """
        body = self.default_body.copy()
        repeated_remote_id = uuid.uuid4().hex
        body['remote_ids'] = [uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex, repeated_remote_id]
        self._create_default_idp(body=body)
        url = self.base_url(suffix=uuid.uuid4().hex)
        body['remote_ids'] = [uuid.uuid4().hex, repeated_remote_id]
        resp = self.put(url, body={'identity_provider': body}, expected_status=http.client.CONFLICT)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Duplicate remote ID', resp_data.get('error', {}).get('message'))

    def test_create_idp_remote_empty(self):
        """Create an IdP with empty remote_ids."""
        keys_to_check = list(self.idp_keys)
        keys_to_check.append('remote_ids')
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        body['remote_ids'] = []
        resp = self._create_default_idp(body=body)
        self.assertValidResponse(resp, 'identity_provider', dummy_validator, keys_to_check=keys_to_check, ref=body)

    def test_create_idp_remote_none(self):
        """Create an IdP with a None remote_ids."""
        keys_to_check = list(self.idp_keys)
        keys_to_check.append('remote_ids')
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        body['remote_ids'] = None
        resp = self._create_default_idp(body=body)
        expected = body.copy()
        expected['remote_ids'] = []
        self.assertValidResponse(resp, 'identity_provider', dummy_validator, keys_to_check=keys_to_check, ref=expected)

    def test_create_idp_authorization_ttl(self):
        keys_to_check = list(self.idp_keys)
        keys_to_check.append('authorization_ttl')
        body = self.default_body.copy()
        body['description'] = uuid.uuid4().hex
        body['authorization_ttl'] = 10080
        resp = self._create_default_idp(body)
        expected = body.copy()
        self.assertValidResponse(resp, 'identity_provider', dummy_validator, keys_to_check=keys_to_check, ref=expected)

    def test_update_idp_remote_ids(self):
        """Update IdP's remote_ids parameter."""
        body = self.default_body.copy()
        body['remote_ids'] = [uuid.uuid4().hex]
        default_resp = self._create_default_idp(body=body)
        default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        idp_id = default_idp.get('id')
        url = self.base_url(suffix=idp_id)
        self.assertIsNotNone(idp_id)
        body['remote_ids'] = [uuid.uuid4().hex, uuid.uuid4().hex]
        body = {'identity_provider': body}
        resp = self.patch(url, body=body)
        updated_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
        body = body['identity_provider']
        self.assertEqual(sorted(body['remote_ids']), sorted(updated_idp.get('remote_ids')))
        resp = self.get(url)
        returned_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
        self.assertEqual(sorted(body['remote_ids']), sorted(returned_idp.get('remote_ids')))

    def test_update_idp_clean_remote_ids(self):
        """Update IdP's remote_ids parameter with an empty list."""
        body = self.default_body.copy()
        body['remote_ids'] = [uuid.uuid4().hex]
        default_resp = self._create_default_idp(body=body)
        default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        idp_id = default_idp.get('id')
        url = self.base_url(suffix=idp_id)
        self.assertIsNotNone(idp_id)
        body['remote_ids'] = []
        body = {'identity_provider': body}
        resp = self.patch(url, body=body)
        updated_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
        body = body['identity_provider']
        self.assertEqual(sorted(body['remote_ids']), sorted(updated_idp.get('remote_ids')))
        resp = self.get(url)
        returned_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
        self.assertEqual(sorted(body['remote_ids']), sorted(returned_idp.get('remote_ids')))

    def test_update_idp_remote_repeated(self):
        """Update an IdentityProvider entity reusing a remote_id.

        A remote_id is the same for both so the second IdP is not
        updated because of the uniqueness of the remote_ids.

        Expect HTTP 409 Conflict code for the latter call.

        """
        body = self.default_body.copy()
        repeated_remote_id = uuid.uuid4().hex
        body['remote_ids'] = [uuid.uuid4().hex, repeated_remote_id]
        self._create_default_idp(body=body)
        body = self.default_body.copy()
        default_resp = self._create_default_idp(body=body)
        default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        idp_id = default_idp.get('id')
        url = self.base_url(suffix=idp_id)
        body['remote_ids'] = [repeated_remote_id]
        resp = self.patch(url, body={'identity_provider': body}, expected_status=http.client.CONFLICT)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Duplicate remote ID', resp_data['error']['message'])

    def test_update_idp_authorization_ttl(self):
        body = self.default_body.copy()
        body['authorization_ttl'] = 10080
        default_resp = self._create_default_idp(body=body)
        default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        idp_id = default_idp.get('id')
        url = self.base_url(suffix=idp_id)
        self.assertIsNotNone(idp_id)
        body['authorization_ttl'] = None
        body = {'identity_provider': body}
        resp = self.patch(url, body=body)
        updated_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
        body = body['identity_provider']
        self.assertEqual(body['authorization_ttl'], updated_idp.get('authorization_ttl'))
        resp = self.get(url)
        returned_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
        self.assertEqual(body['authorization_ttl'], returned_idp.get('authorization_ttl'))

    def test_list_head_idps(self, iterations=5):
        """List all available IdentityProviders.

        This test collects ids of created IdPs and
        intersects it with the list of all available IdPs.
        List of all IdPs can be a superset of IdPs created in this test,
        because other tests also create IdPs.

        """

        def get_id(resp):
            r = self._fetch_attribute_from_response(resp, 'identity_provider')
            return r.get('id')
        ids = []
        for _ in range(iterations):
            id = get_id(self._create_default_idp())
            ids.append(id)
        ids = set(ids)
        keys_to_check = self.idp_keys
        keys_to_check.append('domain_id')
        url = self.base_url()
        resp = self.get(url)
        self.assertValidListResponse(resp, 'identity_providers', dummy_validator, keys_to_check=keys_to_check)
        entities = self._fetch_attribute_from_response(resp, 'identity_providers')
        entities_ids = set([e['id'] for e in entities])
        ids_intersection = entities_ids.intersection(ids)
        self.assertEqual(ids_intersection, ids)
        self.head(url, expected_status=http.client.OK)

    def test_filter_list_head_idp_by_id(self):

        def get_id(resp):
            r = self._fetch_attribute_from_response(resp, 'identity_provider')
            return r.get('id')
        idp1_id = get_id(self._create_default_idp())
        idp2_id = get_id(self._create_default_idp())
        url = self.base_url()
        resp = self.get(url)
        entities = self._fetch_attribute_from_response(resp, 'identity_providers')
        entities_ids = [e['id'] for e in entities]
        self.assertCountEqual(entities_ids, [idp1_id, idp2_id])
        url = self.base_url() + '?id=' + idp1_id
        resp = self.get(url)
        filtered_service_list = resp.json['identity_providers']
        self.assertThat(filtered_service_list, matchers.HasLength(1))
        self.assertEqual(idp1_id, filtered_service_list[0].get('id'))
        self.head(url, expected_status=http.client.OK)

    def test_filter_list_head_idp_by_enabled(self):

        def get_id(resp):
            r = self._fetch_attribute_from_response(resp, 'identity_provider')
            return r.get('id')
        idp1_id = get_id(self._create_default_idp())
        body = self.default_body.copy()
        body['enabled'] = False
        idp2_id = get_id(self._create_default_idp(body=body))
        url = self.base_url()
        resp = self.get(url)
        entities = self._fetch_attribute_from_response(resp, 'identity_providers')
        entities_ids = [e['id'] for e in entities]
        self.assertCountEqual(entities_ids, [idp1_id, idp2_id])
        url = self.base_url() + '?enabled=True'
        resp = self.get(url)
        filtered_service_list = resp.json['identity_providers']
        self.assertThat(filtered_service_list, matchers.HasLength(1))
        self.assertEqual(idp1_id, filtered_service_list[0].get('id'))
        self.head(url, expected_status=http.client.OK)

    def test_check_idp_uniqueness(self):
        """Add same IdP twice.

        Expect HTTP 409 Conflict code for the latter call.

        """
        url = self.base_url(suffix=uuid.uuid4().hex)
        body = self._http_idp_input()
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        body['domain_id'] = domain['id']
        self.put(url, body={'identity_provider': body}, expected_status=http.client.CREATED)
        resp = self.put(url, body={'identity_provider': body}, expected_status=http.client.CONFLICT)
        resp_data = jsonutils.loads(resp.body)
        self.assertIn('Duplicate entry', resp_data.get('error', {}).get('message'))

    def test_get_head_idp(self):
        """Create and later fetch IdP."""
        body = self._http_idp_input()
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        body['domain_id'] = domain['id']
        default_resp = self._create_default_idp(body=body)
        default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        idp_id = default_idp.get('id')
        url = self.base_url(suffix=idp_id)
        resp = self.get(url)
        body_keys = list(body)
        self.assertValidResponse(resp, 'identity_provider', dummy_validator, keys_to_check=body_keys, ref=body)
        self.head(url, expected_status=http.client.OK)

    def test_get_nonexisting_idp(self):
        """Fetch nonexisting IdP entity.

        Expected HTTP 404 Not Found status code.

        """
        idp_id = uuid.uuid4().hex
        self.assertIsNotNone(idp_id)
        url = self.base_url(suffix=idp_id)
        self.get(url, expected_status=http.client.NOT_FOUND)

    def test_delete_existing_idp(self):
        """Create and later delete IdP.

        Expect HTTP 404 Not Found for the GET IdP call.
        """
        default_resp = self._create_default_idp()
        default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        idp_id = default_idp.get('id')
        self.assertIsNotNone(idp_id)
        url = self.base_url(suffix=idp_id)
        self.delete(url)
        self.get(url, expected_status=http.client.NOT_FOUND)

    def test_delete_idp_also_deletes_assigned_protocols(self):
        """Deleting an IdP will delete its assigned protocol."""
        default_resp = self._create_default_idp()
        default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        idp_id = default_idp['id']
        protocol_id = uuid.uuid4().hex
        url = self.base_url(suffix='%(idp_id)s/protocols/%(protocol_id)s')
        idp_url = self.base_url(suffix=idp_id)
        kwargs = {'expected_status': http.client.CREATED}
        resp, idp_id, proto = self._assign_protocol_to_idp(url=url, idp_id=idp_id, proto=protocol_id, **kwargs)
        self.assertEqual(1, len(PROVIDERS.federation_api.list_protocols(idp_id)))
        self.delete(idp_url)
        self.get(idp_url, expected_status=http.client.NOT_FOUND)
        self.assertEqual(0, len(PROVIDERS.federation_api.list_protocols(idp_id)))

    def test_delete_nonexisting_idp(self):
        """Delete nonexisting IdP.

        Expect HTTP 404 Not Found for the GET IdP call.
        """
        idp_id = uuid.uuid4().hex
        url = self.base_url(suffix=idp_id)
        self.delete(url, expected_status=http.client.NOT_FOUND)

    def test_update_idp_mutable_attributes(self):
        """Update IdP's mutable parameters."""
        default_resp = self._create_default_idp()
        default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        idp_id = default_idp.get('id')
        url = self.base_url(suffix=idp_id)
        self.assertIsNotNone(idp_id)
        _enabled = not default_idp.get('enabled')
        body = {'remote_ids': [uuid.uuid4().hex, uuid.uuid4().hex], 'description': uuid.uuid4().hex, 'enabled': _enabled}
        body = {'identity_provider': body}
        resp = self.patch(url, body=body)
        updated_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
        body = body['identity_provider']
        for key in body.keys():
            if isinstance(body[key], list):
                self.assertEqual(sorted(body[key]), sorted(updated_idp.get(key)))
            else:
                self.assertEqual(body[key], updated_idp.get(key))
        resp = self.get(url)
        updated_idp = self._fetch_attribute_from_response(resp, 'identity_provider')
        for key in body.keys():
            if isinstance(body[key], list):
                self.assertEqual(sorted(body[key]), sorted(updated_idp.get(key)))
            else:
                self.assertEqual(body[key], updated_idp.get(key))

    def test_update_idp_immutable_attributes(self):
        """Update IdP's immutable parameters.

        Expect HTTP BAD REQUEST.

        """
        default_resp = self._create_default_idp()
        default_idp = self._fetch_attribute_from_response(default_resp, 'identity_provider')
        idp_id = default_idp.get('id')
        self.assertIsNotNone(idp_id)
        body = self._http_idp_input()
        body['id'] = uuid.uuid4().hex
        body['protocols'] = [uuid.uuid4().hex, uuid.uuid4().hex]
        url = self.base_url(suffix=idp_id)
        self.patch(url, body={'identity_provider': body}, expected_status=http.client.BAD_REQUEST)

    def test_update_nonexistent_idp(self):
        """Update nonexistent IdP.

        Expect HTTP 404 Not Found code.

        """
        idp_id = uuid.uuid4().hex
        url = self.base_url(suffix=idp_id)
        body = self._http_idp_input()
        body['enabled'] = False
        body = {'identity_provider': body}
        self.patch(url, body=body, expected_status=http.client.NOT_FOUND)

    def test_assign_protocol_to_idp(self):
        """Assign a protocol to existing IdP."""
        self._assign_protocol_to_idp(expected_status=http.client.CREATED)

    def test_protocol_composite_pk(self):
        """Test that Keystone can add two entities.

        The entities have identical names, however, attached to different
        IdPs.

        1. Add IdP and assign it protocol with predefined name
        2. Add another IdP and assign it a protocol with same name.

        Expect HTTP 201 code

        """
        url = self.base_url(suffix='%(idp_id)s/protocols/%(protocol_id)s')
        kwargs = {'expected_status': http.client.CREATED}
        self._assign_protocol_to_idp(proto='saml2', url=url, **kwargs)
        self._assign_protocol_to_idp(proto='saml2', url=url, **kwargs)

    def test_protocol_idp_pk_uniqueness(self):
        """Test whether Keystone checks for unique idp/protocol values.

        Add same protocol twice, expect Keystone to reject a latter call and
        return HTTP 409 Conflict code.

        """
        url = self.base_url(suffix='%(idp_id)s/protocols/%(protocol_id)s')
        kwargs = {'expected_status': http.client.CREATED}
        resp, idp_id, proto = self._assign_protocol_to_idp(proto='saml2', url=url, **kwargs)
        kwargs = {'expected_status': http.client.CONFLICT}
        self._assign_protocol_to_idp(idp_id=idp_id, proto='saml2', validate=False, url=url, **kwargs)

    def test_assign_protocol_to_nonexistent_idp(self):
        """Assign protocol to IdP that doesn't exist.

        Expect HTTP 404 Not Found code.

        """
        idp_id = uuid.uuid4().hex
        kwargs = {'expected_status': http.client.NOT_FOUND}
        self._assign_protocol_to_idp(proto='saml2', idp_id=idp_id, validate=False, **kwargs)

    def test_crud_protocol_without_protocol_id_in_url(self):
        idp_id, _ = self._create_and_decapsulate_response()
        mapping_id = uuid.uuid4().hex
        self._create_mapping(mapping_id=mapping_id)
        protocol = {'id': uuid.uuid4().hex, 'mapping_id': mapping_id}
        with self.test_client() as c:
            token = self.get_scoped_token()
            c.delete('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols' % {'idp_id': idp_id}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
            c.patch('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols/' % {'idp_id': idp_id}, json={'protocol': protocol}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
            c.put('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols' % {'idp_id': idp_id}, json={'protocol': protocol}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
            c.delete('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols/' % {'idp_id': idp_id}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
            c.patch('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols/' % {'idp_id': idp_id}, json={'protocol': protocol}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)
            c.put('/v3/OS-FEDERATION/identity_providers/%(idp_id)s/protocols/' % {'idp_id': idp_id}, json={'protocol': protocol}, headers={'X-Auth-Token': token}, expected_status_code=http.client.METHOD_NOT_ALLOWED)

    def test_get_head_protocol(self):
        """Create and later fetch protocol tied to IdP."""
        resp, idp_id, proto = self._assign_protocol_to_idp(expected_status=http.client.CREATED)
        proto_id = self._fetch_attribute_from_response(resp, 'protocol')['id']
        url = '%s/protocols/%s' % (idp_id, proto_id)
        url = self.base_url(suffix=url)
        resp = self.get(url)
        reference = {'id': proto_id}
        reference_keys = list(reference)
        self.assertValidResponse(resp, 'protocol', dummy_validator, keys_to_check=reference_keys, ref=reference)
        self.head(url, expected_status=http.client.OK)

    def test_list_head_protocols(self):
        """Create set of protocols and later list them.

        Compare input and output id sets.

        """
        resp, idp_id, proto = self._assign_protocol_to_idp(expected_status=http.client.CREATED)
        iterations = random.randint(0, 16)
        protocol_ids = []
        for _ in range(iterations):
            resp, _, proto = self._assign_protocol_to_idp(idp_id=idp_id, expected_status=http.client.CREATED)
            proto_id = self._fetch_attribute_from_response(resp, 'protocol')
            proto_id = proto_id['id']
            protocol_ids.append(proto_id)
        url = '%s/protocols' % idp_id
        url = self.base_url(suffix=url)
        resp = self.get(url)
        self.assertValidListResponse(resp, 'protocols', dummy_validator, keys_to_check=['id'])
        entities = self._fetch_attribute_from_response(resp, 'protocols')
        entities = set([entity['id'] for entity in entities])
        protocols_intersection = entities.intersection(protocol_ids)
        self.assertEqual(protocols_intersection, set(protocol_ids))
        self.head(url, expected_status=http.client.OK)

    def test_update_protocols_attribute(self):
        """Update protocol's attribute."""
        resp, idp_id, proto = self._assign_protocol_to_idp(expected_status=http.client.CREATED)
        new_mapping_id = uuid.uuid4().hex
        self._create_mapping(mapping_id=new_mapping_id)
        url = '%s/protocols/%s' % (idp_id, proto)
        url = self.base_url(suffix=url)
        body = {'mapping_id': new_mapping_id}
        resp = self.patch(url, body={'protocol': body})
        self.assertValidResponse(resp, 'protocol', dummy_validator, keys_to_check=['id', 'mapping_id'], ref={'id': proto, 'mapping_id': new_mapping_id})

    def test_delete_protocol(self):
        """Delete protocol.

        Expect HTTP 404 Not Found code for the GET call after the protocol is
        deleted.

        """
        url = self.base_url(suffix='%(idp_id)s/protocols/%(protocol_id)s')
        resp, idp_id, proto = self._assign_protocol_to_idp(expected_status=http.client.CREATED)
        url = url % {'idp_id': idp_id, 'protocol_id': proto}
        self.delete(url)
        self.get(url, expected_status=http.client.NOT_FOUND)