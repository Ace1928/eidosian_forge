from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
def _session_client(**kwargs):
    return http.SessionClient(os_ironic_api_version='1.6', api_version_select_state='default', max_retries=5, retry_interval=2, auth=None, interface='publicURL', service_type='baremetal', region_name='', endpoint_override='http://%s:%s' % (DEFAULT_HOST, DEFAULT_PORT), **kwargs)