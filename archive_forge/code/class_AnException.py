from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
class AnException(Exception):
    pass