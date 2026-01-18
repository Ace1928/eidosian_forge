import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def _assert_saved_fields(self, expected, actual):
    for k in expected.keys():
        self.assertEqual(expected[k], actual[k])