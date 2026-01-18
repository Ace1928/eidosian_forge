import copy
from unittest import mock
from oslo_messaging._drivers import common as rpc_common
from oslo_utils import reflection
from heat.common import exception
from heat.common import identifier
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def _test_engine_api(self, method, rpc_method, **kwargs):
    ctxt = utils.dummy_context()
    expected_retval = 'foo' if method == 'call' else None
    kwargs.pop('version', None)
    if 'expected_message' in kwargs:
        expected_message = kwargs['expected_message']
        del kwargs['expected_message']
    else:
        expected_message = self.rpcapi.make_msg(method, **kwargs)
    cast_and_call = ['delete_stack']
    if method in cast_and_call:
        kwargs['cast'] = rpc_method != 'call'
    with mock.patch.object(self.rpcapi, rpc_method) as mock_rpc_method:
        mock_rpc_method.return_value = expected_retval
        retval = getattr(self.rpcapi, method)(ctxt, **kwargs)
        self.assertEqual(expected_retval, retval)
        expected_args = [ctxt, expected_message, mock.ANY]
        actual_args, _ = mock_rpc_method.call_args
        for expected_arg, actual_arg in zip(expected_args, actual_args):
            self.assertEqual(expected_arg, actual_arg)