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