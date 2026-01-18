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