from unittest import mock
import uuid
import fixtures
import flask
from flask import blueprints
import flask_restful
from oslo_policy import policy
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import rest
def _enforce_mock_func(credentials, action, target, do_raise=True):
    assertIn('target.domain.id', target)
    assertEq(target['target.domain.id'], ref_uuid)