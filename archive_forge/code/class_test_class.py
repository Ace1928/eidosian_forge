from unittest import mock
from oslotest import base as test_base
from oslo_log import helpers
class test_class(object):

    @staticmethod
    @helpers.log_method_call
    def test_staticmethod(arg1, arg2, arg3, *args, **kwargs):
        pass