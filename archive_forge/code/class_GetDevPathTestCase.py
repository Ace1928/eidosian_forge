import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@ddt.ddt
class GetDevPathTestCase(base.TestCase):
    """Test the get_dev_path method."""

    @ddt.data({'con_props': {}, 'dev_info': {'path': '/dev/sda'}}, {'con_props': {}, 'dev_info': {'path': b'/dev/sda'}}, {'con_props': None, 'dev_info': {'path': '/dev/sda'}}, {'con_props': None, 'dev_info': {'path': b'/dev/sda'}}, {'con_props': {'device_path': b'/dev/sdb'}, 'dev_info': {'path': '/dev/sda'}}, {'con_props': {'device_path': '/dev/sdb'}, 'dev_info': {'path': b'/dev/sda'}})
    @ddt.unpack
    def test_get_dev_path_device_info(self, con_props, dev_info):
        self.assertEqual('/dev/sda', utils.get_dev_path(con_props, dev_info))

    @ddt.data({'con_props': {'device_path': '/dev/sda'}, 'dev_info': {'path': None}}, {'con_props': {'device_path': b'/dev/sda'}, 'dev_info': {'path': None}}, {'con_props': {'device_path': '/dev/sda'}, 'dev_info': {'path': ''}}, {'con_props': {'device_path': b'/dev/sda'}, 'dev_info': {'path': ''}}, {'con_props': {'device_path': '/dev/sda'}, 'dev_info': {}}, {'con_props': {'device_path': b'/dev/sda'}, 'dev_info': {}}, {'con_props': {'device_path': '/dev/sda'}, 'dev_info': None}, {'con_props': {'device_path': b'/dev/sda'}, 'dev_info': None})
    @ddt.unpack
    def test_get_dev_path_conn_props(self, con_props, dev_info):
        self.assertEqual('/dev/sda', utils.get_dev_path(con_props, dev_info))

    @ddt.data({'con_props': {'device_path': ''}, 'dev_info': {'path': None}}, {'con_props': {'device_path': None}, 'dev_info': {'path': ''}}, {'con_props': {}, 'dev_info': {}}, {'con_props': {}, 'dev_info': None})
    @ddt.unpack
    def test_get_dev_path_no_path(self, con_props, dev_info):
        self.assertEqual('', utils.get_dev_path(con_props, dev_info))