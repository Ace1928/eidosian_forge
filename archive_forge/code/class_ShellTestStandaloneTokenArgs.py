import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
class ShellTestStandaloneTokenArgs(StandaloneTokenMixin, ShellTestNoMoxBase):

    def test_commandline_args_passed_to_requests(self):
        """Check that we have sent the proper arguments to requests."""
        self.register_keystone_auth_fixture()
        resp_dict = {'stacks': [{'id': '1', 'stack_name': 'teststack', 'stack_owner': 'testowner', 'project': 'testproject', 'stack_status': 'CREATE_COMPLETE', 'creation_time': '2014-10-15T01:58:47Z'}]}
        self.requests.get('http://no.where/stacks', status_code=200, headers={'Content-Type': 'application/json'}, json=resp_dict)
        list_text = self.shell('stack-list')
        required = ['id', 'stack_status', 'creation_time', 'teststack', '1', 'CREATE_COMPLETE']
        for r in required:
            self.assertRegex(list_text, r)
        self.assertNotRegex(list_text, 'parent')