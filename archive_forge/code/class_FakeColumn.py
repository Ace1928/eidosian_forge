from unittest import mock
from neutron_lib.objects import utils as obj_utils
from neutron_lib.tests import _base as base
class FakeColumn(object):

    def __init__(self, column):
        self.column = column

    def __ne__(self, value):
        return [item for item in self.column if item != value]