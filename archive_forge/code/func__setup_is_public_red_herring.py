import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def _setup_is_public_red_herring(self):
    values = {'name': 'Red Herring', 'owner': self.tenant1, 'visibility': 'shared', 'properties': {'is_public': 'silly'}}
    fixture = build_image_fixture(**values)
    self.db_api.image_create(self.admin_context, fixture)