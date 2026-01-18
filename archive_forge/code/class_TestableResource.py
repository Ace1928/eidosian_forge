import copy
from unittest import mock
import testtools
from ironicclient.common import base
from ironicclient import exc
from ironicclient.tests.unit import utils
class TestableResource(base.Resource):

    def __repr__(self):
        return '<TestableResource %s>' % self._info