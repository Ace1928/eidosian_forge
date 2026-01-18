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
class K2KServiceCatalogTests(test_v3.RestfulTestCase):
    SP1 = 'SP1'
    SP2 = 'SP2'
    SP3 = 'SP3'

    def setUp(self):
        super(K2KServiceCatalogTests, self).setUp()
        sp = core.new_service_provider_ref()
        PROVIDERS.federation_api.create_sp(self.SP1, sp)
        self.sp_alpha = {self.SP1: sp}
        sp = core.new_service_provider_ref()
        PROVIDERS.federation_api.create_sp(self.SP2, sp)
        self.sp_beta = {self.SP2: sp}
        sp = core.new_service_provider_ref()
        PROVIDERS.federation_api.create_sp(self.SP3, sp)
        self.sp_gamma = {self.SP3: sp}

    def sp_response(self, id, ref):
        ref.pop('enabled')
        ref.pop('description')
        ref.pop('relay_state_prefix')
        ref['id'] = id
        return ref

    def _validate_service_providers(self, token, ref):
        token_data = token['token']
        self.assertIn('service_providers', token_data)
        self.assertIsNotNone(token_data['service_providers'])
        service_providers = token_data.get('service_providers')
        self.assertEqual(len(ref), len(service_providers))
        for entity in service_providers:
            id = entity.get('id')
            ref_entity = self.sp_response(id, ref.get(id))
            self.assertDictEqual(entity, ref_entity)

    def test_service_providers_in_token(self):
        """Check if service providers are listed in service catalog."""
        model = token_model.TokenModel()
        model.user_id = self.user_id
        model.methods = ['password']
        token = render_token.render_token_response_from_model(model)
        ref = {}
        for r in (self.sp_alpha, self.sp_beta, self.sp_gamma):
            ref.update(r)
        self._validate_service_providers(token, ref)

    def test_service_provides_in_token_disabled_sp(self):
        """Test behaviour with disabled service providers.

        Disabled service providers should not be listed in the service
        catalog.

        """
        sp_ref = {'enabled': False}
        PROVIDERS.federation_api.update_sp(self.SP1, sp_ref)
        model = token_model.TokenModel()
        model.user_id = self.user_id
        model.methods = ['password']
        token = render_token.render_token_response_from_model(model)
        ref = {}
        for r in (self.sp_beta, self.sp_gamma):
            ref.update(r)
        self._validate_service_providers(token, ref)

    def test_no_service_providers_in_token(self):
        """Test service catalog with disabled service providers.

        There should be no entry ``service_providers`` in the catalog.
        Test passes providing no attribute was raised.

        """
        sp_ref = {'enabled': False}
        for sp in (self.SP1, self.SP2, self.SP3):
            PROVIDERS.federation_api.update_sp(sp, sp_ref)
        model = token_model.TokenModel()
        model.user_id = self.user_id
        model.methods = ['password']
        token = render_token.render_token_response_from_model(model)
        self.assertNotIn('service_providers', token['token'], message='Expected Service Catalog not to have service_providers')