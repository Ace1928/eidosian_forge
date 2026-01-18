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
def _setup_enforcer_object(self):
    self.enforcer = rbac_enforcer.enforcer.RBACEnforcer()
    self.cleanup_instance('enforcer')

    def register_new_rules(enforcer):
        rules = self._testing_policy_rules()
        enforcer.register_defaults(rules)
    self.useFixture(fixtures.MockPatchObject(self.enforcer, 'register_rules', register_new_rules))
    original_actions = rbac_enforcer.enforcer._POSSIBLE_TARGET_ACTIONS
    rbac_enforcer.enforcer._POSSIBLE_TARGET_ACTIONS = frozenset([rule.name for rule in self._testing_policy_rules()])
    self.addCleanup(setattr, rbac_enforcer.enforcer, '_POSSIBLE_TARGET_ACTIONS', original_actions)
    self.enforcer._reset()