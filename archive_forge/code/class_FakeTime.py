from unittest import mock
from oslo_db import exception as db_exc
import osprofiler
import sqlalchemy
from sqlalchemy.orm import exc
import testtools
from neutron_lib.db import api as db_api
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _base
class FakeTime(object):

    def __init__(self):
        self.counter = 0

    def sleep(self, t):
        self.counter += t