import collections
import ctypes
from unittest import mock
import ddt
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
@mock.patch.object(iscsi_utils, '_get_items_from_buff')
def _test_ensure_buff_decorator(self, mock_get_items, required_buff_sz=None, returned_element_count=None, parse_output=False):
    insufficient_buff_exc = exceptions.Win32Exception(message='fake_err_msg', error_code=w_const.ERROR_INSUFFICIENT_BUFFER)
    func_requests_buff_sz = required_buff_sz is not None
    struct_type = ctypes.c_uint
    decorator_args = dict(struct_type=struct_type, parse_output=parse_output, func_requests_buff_sz=func_requests_buff_sz)
    func_side_effect = mock.Mock(side_effect=(insufficient_buff_exc, None))
    fake_func = self._get_fake_iscsi_utils_getter_func(returned_element_count=returned_element_count, required_buff_sz=required_buff_sz, func_side_effect=func_side_effect, decorator_args=decorator_args)
    ret_val = fake_func(self._initiator, fake_arg=mock.sentinel.arg)
    if parse_output:
        self.assertEqual(mock_get_items.return_value, ret_val)
    else:
        self.assertEqual(mock.sentinel.ret_val, ret_val)
    first_call_args_dict = func_side_effect.call_args_list[0][1]
    self.assertIsInstance(first_call_args_dict['buff'], ctypes.POINTER(struct_type))
    self.assertEqual(first_call_args_dict['buff_size_val'], 0)
    self.assertEqual(first_call_args_dict['element_count_val'], 0)
    second_call_args_dict = func_side_effect.call_args_list[1][1]
    self.assertIsInstance(second_call_args_dict['buff'], ctypes.POINTER(struct_type))
    self.assertEqual(second_call_args_dict['buff_size_val'], required_buff_sz or 0)
    self.assertEqual(second_call_args_dict['element_count_val'], returned_element_count or 0)