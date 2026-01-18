import flask
import uuid
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.auth.plugins import mapped
import keystone.conf
from keystone import exception
from keystone.federation import utils as mapping_utils
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from unittest import mock
def _pull_mapping_rules_from_the_database(self):
    return jsonutils.loads(jsonutils.dumps(mapping_fixtures.MAPPING_UNICODE))