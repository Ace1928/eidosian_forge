import copy
import functools
import random
import http.client
from oslo_serialization import jsonutils
from testtools import matchers as tt_matchers
import webob
from keystone.api import discovery
from keystone.common import json_home
from keystone.tests import unit
def _paste_in_port(self, response, port):
    for link in response['links']:
        if link['rel'] == 'self':
            link['href'] = port