import argparse
import io
import json
import re
import sys
from unittest import mock
import ddt
import fixtures
import keystoneauth1.exceptions as ks_exc
from keystoneauth1.exceptions import DiscoveryFailure
from keystoneauth1.identity.generic.password import Password as ks_password
from keystoneauth1 import session
import requests_mock
from testtools import matchers
import cinderclient
from cinderclient import api_versions
from cinderclient.contrib import noauth
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit import fake_actions_module
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
class ShellUtilsTest(utils.TestCase):

    @mock.patch.object(cinderclient.shell_utils, 'print_dict')
    def test_print_volume_image(self, mock_print_dict):
        response = {'os-volume_upload_image': {'name': 'myimg1'}}
        image_resp_tuple = (202, response)
        cinderclient.shell_utils.print_volume_image(image_resp_tuple)
        response = {'os-volume_upload_image': {'name': 'myimg2', 'volume_type': None}}
        image_resp_tuple = (202, response)
        cinderclient.shell_utils.print_volume_image(image_resp_tuple)
        response = {'os-volume_upload_image': {'name': 'myimg3', 'volume_type': {'id': '1234', 'name': 'sometype'}}}
        image_resp_tuple = (202, response)
        cinderclient.shell_utils.print_volume_image(image_resp_tuple)
        mock_print_dict.assert_has_calls((mock.call({'name': 'myimg1'}), mock.call({'name': 'myimg2', 'volume_type': None}), mock.call({'name': 'myimg3', 'volume_type': 'sometype'})))