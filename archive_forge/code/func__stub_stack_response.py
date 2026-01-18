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
def _stub_stack_response(self, stack_id, action='CREATE', status='IN_PROGRESS'):
    resp_dict = {'stack': {'id': stack_id.split('/')[1], 'stack_name': stack_id.split('/')[0], 'stack_status': '%s_%s' % (action, status), 'creation_time': '2014-01-06T16:14:00Z'}}
    self.mock_request_get('/stacks/teststack/1', resp_dict)