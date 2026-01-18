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
def _issue_unscoped_token(self, idp=None, assertion='EMPLOYEE_ASSERTION', environment=None):
    environment = environment or {}
    environment.update(getattr(mapping_fixtures, assertion))
    with self.make_request(environ=environment):
        if idp is None:
            idp = self.IDP
        r = authentication.federated_authenticate_for_token(protocol_id=self.PROTOCOL, identity_provider=idp)
    return r