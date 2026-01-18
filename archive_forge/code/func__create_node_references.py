import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def _create_node_references(self):
    self.node_references = [_db_node_reference_fixture(NODE_REFERENCE_ID_1, 'node_url_1'), _db_node_reference_fixture(NODE_REFERENCE_ID_2, 'node_url_2')]
    [self.db.node_reference_create(None, node_reference['node_reference_url'], node_reference_id=node_reference['node_reference_id']) for node_reference in self.node_references]