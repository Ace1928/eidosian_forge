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
def _assert_last_notify(self, action, identity_provider, protocol, user_id=None):
    self.assertTrue(self._notifications)
    note = self._notifications[-1]
    if user_id:
        self.assertEqual(note['user_id'], user_id)
    self.assertEqual(note['action'], action)
    self.assertEqual(note['identity_provider'], identity_provider)
    self.assertEqual(note['protocol'], protocol)
    self.assertTrue(note['send_notification_called'])