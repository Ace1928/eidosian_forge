import uuid
import fixtures
import flask
import flask_restful
import functools
from oslo_policy import policy
from oslo_serialization import jsonutils
from testtools import matchers
from keystone.common import context
from keystone.common import json_home
from keystone.common import rbac_enforcer
import keystone.conf
from keystone import exception
from keystone.server.flask import common as flask_common
from keystone.server.flask.request_processing import json_body
from keystone.tests.unit import rest
class _TestRestfulAPI(flask_common.APIBase):
    _name = 'test_api_base'
    _import_name = __name__
    resources = []
    resource_mapping = []

    def __init__(self, *args, **kwargs):
        self.resource_mapping = kwargs.pop('resource_mapping', [])
        self.resources = kwargs.pop('resources', [_TestResourceWithCollectionInfo])
        super(_TestRestfulAPI, self).__init__(*args, **kwargs)