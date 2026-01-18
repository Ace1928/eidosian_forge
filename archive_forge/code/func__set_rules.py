import os
import subprocess
from unittest import mock
import uuid
from oslo_policy import policy as common_policy
from keystone.common import policies
from keystone.common.rbac_enforcer import policy
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def _set_rules(self):
    these_rules = common_policy.Rules.from_dict(self.rules)
    policy._ENFORCER._enforcer.set_rules(these_rules)