import json
from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
import requests
from heat.api.aws import ec2token
from heat.api.aws import exception
from heat.common import wsgi
from heat.tests import common
from heat.tests import utils
class DummyHTTPResponse(object):
    text = response
    headers = {'X-Subject-Token': 123}

    def json(self):
        return json.loads(self.text)