import copy
import datetime
import re
from unittest import mock
from urllib import parse
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client as base_client
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils
from novaclient.v2 import client
def _return_server_group(self):
    if self.api_version < api_versions.APIVersion('2.64'):
        r = {'server_group': self.get_os_server_groups()[2]['server_groups'][0]}
    else:
        r = {'members': [], 'id': '2cbd51f4-fafe-4cdb-801b-cf913a6f288b', 'server_group': {'name': 'ig1', 'policy': 'anti-affinity', 'rules': {'max_server_per_host': 3}}}
    return (200, {}, r)