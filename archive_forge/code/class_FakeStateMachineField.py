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
class FakeStateMachineField(fields.StateMachine):
    ACTIVE = 'ACTIVE'
    PENDING = 'PENDING'
    ERROR = 'ERROR'
    ALLOWED_TRANSITIONS = {ACTIVE: {PENDING, ERROR}, PENDING: {ACTIVE, ERROR}, ERROR: {PENDING}}
    _TYPES = (ACTIVE, PENDING, ERROR)

    def __init__(self, **kwargs):
        super(FakeStateMachineField, self).__init__(self._TYPES, **kwargs)