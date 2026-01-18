import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
class FakeEnum(fields.Enum):
    FROG = 'frog'
    PLATYPUS = 'platypus'
    ALLIGATOR = 'alligator'
    ALL = (FROG, PLATYPUS, ALLIGATOR)

    def __init__(self, **kwargs):
        super(FakeEnum, self).__init__(valid_values=FakeEnum.ALL, **kwargs)