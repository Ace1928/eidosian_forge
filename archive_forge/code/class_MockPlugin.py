import functools
from unittest import mock
import uuid
from keystoneauth1 import fixture
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth import base
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit import utils
class MockPlugin(base.BaseAuthPlugin):
    INT_DESC = 'test int'
    FLOAT_DESC = 'test float'
    BOOL_DESC = 'test bool'
    STR_DESC = 'test str'
    STR_DEFAULT = uuid.uuid4().hex

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getitem__(self, key):
        """Get the data of the key."""
        return self._data[key]

    def get_token(self, *args, **kwargs):
        return 'aToken'

    def get_endpoint(self, *args, **kwargs):
        return 'http://test'

    @classmethod
    def get_options(cls):
        return [cfg.IntOpt('a-int', default='3', help=cls.INT_DESC), cfg.BoolOpt('a-bool', help=cls.BOOL_DESC), cfg.FloatOpt('a-float', help=cls.FLOAT_DESC), cfg.StrOpt('a-str', help=cls.STR_DESC, default=cls.STR_DEFAULT)]