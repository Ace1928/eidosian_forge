import copy
import os
import os.path
from glance.common import config
from glance.common import exception
from glance import context
from glance.db.sqlalchemy import metadata
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class MetadefDriverTests(MetadefNamespaceTests, MetadefResourceTypeTests, MetadefResourceTypeAssociationTests, MetadefPropertyTests, MetadefObjectTests, MetadefTagTests, MetadefLoadUnloadTests):
    pass