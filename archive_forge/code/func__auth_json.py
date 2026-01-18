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
def _auth_json(self):
    return {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': self.user_req_admin['name'], 'password': self.user_req_admin['password'], 'domain': {'id': self.user_req_admin['domain_id']}}}}, 'scope': {'project': {'id': self.project_service['id']}}}}