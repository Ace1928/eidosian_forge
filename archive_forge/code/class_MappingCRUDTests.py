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
class MappingCRUDTests(test_v3.RestfulTestCase):
    """A class for testing CRUD operations for Mappings."""
    MAPPING_URL = '/OS-FEDERATION/mappings/'

    def assertValidMappingListResponse(self, resp, *args, **kwargs):
        return self.assertValidListResponse(resp, 'mappings', self.assertValidMapping, *args, keys_to_check=[], **kwargs)

    def assertValidMappingResponse(self, resp, *args, **kwargs):
        return self.assertValidResponse(resp, 'mapping', self.assertValidMapping, *args, keys_to_check=[], **kwargs)

    def assertValidMapping(self, entity, ref=None):
        self.assertIsNotNone(entity.get('id'))
        self.assertIsNotNone(entity.get('rules'))
        if ref:
            self.assertEqual(entity['rules'], ref['rules'])
        return entity

    def _create_default_mapping_entry(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        resp = self.put(url, body={'mapping': mapping_fixtures.MAPPING_LARGE}, expected_status=http.client.CREATED)
        return resp

    def _get_id_from_response(self, resp):
        r = resp.result.get('mapping')
        return r.get('id')

    def test_mapping_create(self):
        resp = self._create_default_mapping_entry()
        self.assertValidMappingResponse(resp, mapping_fixtures.MAPPING_LARGE)

    def test_mapping_list_head(self):
        url = self.MAPPING_URL
        self._create_default_mapping_entry()
        resp = self.get(url)
        entities = resp.result.get('mappings')
        self.assertIsNotNone(entities)
        self.assertResponseStatus(resp, http.client.OK)
        self.assertValidListLinks(resp.result.get('links'))
        self.assertEqual(1, len(entities))
        self.head(url, expected_status=http.client.OK)

    def test_mapping_delete(self):
        url = self.MAPPING_URL + '%(mapping_id)s'
        resp = self._create_default_mapping_entry()
        mapping_id = self._get_id_from_response(resp)
        url = url % {'mapping_id': str(mapping_id)}
        resp = self.delete(url)
        self.assertResponseStatus(resp, http.client.NO_CONTENT)
        self.get(url, expected_status=http.client.NOT_FOUND)

    def test_mapping_get_head(self):
        url = self.MAPPING_URL + '%(mapping_id)s'
        resp = self._create_default_mapping_entry()
        mapping_id = self._get_id_from_response(resp)
        url = url % {'mapping_id': mapping_id}
        resp = self.get(url)
        self.assertValidMappingResponse(resp, mapping_fixtures.MAPPING_LARGE)
        self.head(url, expected_status=http.client.OK)

    def test_mapping_update(self):
        url = self.MAPPING_URL + '%(mapping_id)s'
        resp = self._create_default_mapping_entry()
        mapping_id = self._get_id_from_response(resp)
        url = url % {'mapping_id': mapping_id}
        resp = self.patch(url, body={'mapping': mapping_fixtures.MAPPING_SMALL})
        self.assertValidMappingResponse(resp, mapping_fixtures.MAPPING_SMALL)
        resp = self.get(url)
        self.assertValidMappingResponse(resp, mapping_fixtures.MAPPING_SMALL)

    def test_delete_mapping_dne(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.delete(url, expected_status=http.client.NOT_FOUND)

    def test_get_mapping_dne(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.get(url, expected_status=http.client.NOT_FOUND)

    def test_create_mapping_bad_requirements(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping_fixtures.MAPPING_BAD_REQ})

    def test_create_mapping_no_rules(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping_fixtures.MAPPING_NO_RULES})

    def test_create_mapping_no_remote_objects(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping_fixtures.MAPPING_NO_REMOTE})

    def test_create_mapping_bad_value(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping_fixtures.MAPPING_BAD_VALUE})

    def test_create_mapping_missing_local(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping_fixtures.MAPPING_MISSING_LOCAL})

    def test_create_mapping_missing_type(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping_fixtures.MAPPING_MISSING_TYPE})

    def test_create_mapping_wrong_type(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping_fixtures.MAPPING_WRONG_TYPE})

    def test_create_mapping_extra_remote_properties_not_any_of(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        mapping = mapping_fixtures.MAPPING_EXTRA_REMOTE_PROPS_NOT_ANY_OF
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping})

    def test_create_mapping_extra_remote_properties_any_one_of(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        mapping = mapping_fixtures.MAPPING_EXTRA_REMOTE_PROPS_ANY_ONE_OF
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping})

    def test_create_mapping_extra_remote_properties_just_type(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        mapping = mapping_fixtures.MAPPING_EXTRA_REMOTE_PROPS_JUST_TYPE
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping})

    def test_create_mapping_empty_map(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': {}})

    def test_create_mapping_extra_rules_properties(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping_fixtures.MAPPING_EXTRA_RULES_PROPS})

    def test_create_mapping_with_blacklist_and_whitelist(self):
        """Test for adding whitelist and blacklist in the rule.

        Server should respond with HTTP 400 Bad Request error upon discovering
        both ``whitelist`` and ``blacklist`` keywords in the same rule.

        """
        url = self.MAPPING_URL + uuid.uuid4().hex
        mapping = mapping_fixtures.MAPPING_GROUPS_WHITELIST_AND_BLACKLIST
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': mapping})

    def test_create_mapping_with_local_user_and_local_domain(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        resp = self.put(url, body={'mapping': mapping_fixtures.MAPPING_LOCAL_USER_LOCAL_DOMAIN}, expected_status=http.client.CREATED)
        self.assertValidMappingResponse(resp, mapping_fixtures.MAPPING_LOCAL_USER_LOCAL_DOMAIN)

    def test_create_mapping_with_ephemeral(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        resp = self.put(url, body={'mapping': mapping_fixtures.MAPPING_EPHEMERAL_USER}, expected_status=http.client.CREATED)
        self.assertValidMappingResponse(resp, mapping_fixtures.MAPPING_EPHEMERAL_USER)

    def test_create_mapping_with_bad_user_type(self):
        url = self.MAPPING_URL + uuid.uuid4().hex
        bad_mapping = copy.deepcopy(mapping_fixtures.MAPPING_EPHEMERAL_USER)
        bad_mapping['rules'][0]['local'][0]['user']['type'] = uuid.uuid4().hex
        self.put(url, expected_status=http.client.BAD_REQUEST, body={'mapping': bad_mapping})

    def test_create_shadow_mapping_without_roles_fails(self):
        """Validate that mappings with projects contain roles when created."""
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, body={'mapping': mapping_fixtures.MAPPING_PROJECTS_WITHOUT_ROLES}, expected_status=http.client.BAD_REQUEST)

    def test_update_shadow_mapping_without_roles_fails(self):
        """Validate that mappings with projects contain roles when updated."""
        url = self.MAPPING_URL + uuid.uuid4().hex
        resp = self.put(url, body={'mapping': mapping_fixtures.MAPPING_PROJECTS}, expected_status=http.client.CREATED)
        self.assertValidMappingResponse(resp, mapping_fixtures.MAPPING_PROJECTS)
        self.patch(url, body={'mapping': mapping_fixtures.MAPPING_PROJECTS_WITHOUT_ROLES}, expected_status=http.client.BAD_REQUEST)

    def test_create_shadow_mapping_without_name_fails(self):
        """Validate project mappings contain the project name when created."""
        url = self.MAPPING_URL + uuid.uuid4().hex
        self.put(url, body={'mapping': mapping_fixtures.MAPPING_PROJECTS_WITHOUT_NAME}, expected_status=http.client.BAD_REQUEST)

    def test_update_shadow_mapping_without_name_fails(self):
        """Validate project mappings contain the project name when updated."""
        url = self.MAPPING_URL + uuid.uuid4().hex
        resp = self.put(url, body={'mapping': mapping_fixtures.MAPPING_PROJECTS}, expected_status=http.client.CREATED)
        self.assertValidMappingResponse(resp, mapping_fixtures.MAPPING_PROJECTS)
        self.patch(url, body={'mapping': mapping_fixtures.MAPPING_PROJECTS_WITHOUT_NAME}, expected_status=http.client.BAD_REQUEST)