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
def _get_default_policy_rules(self):
    """Return a dictionary of all in-code policies.

        All policies have a default value that is maintained in code.
        This method returns a dictionary containing all default policies.
        """
    rules = dict()
    for rule in policies.list_rules():
        rules[rule.name] = rule.check_str
    return rules