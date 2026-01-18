import base64
import hashlib
import hmac
import flask
import http.client
from oslo_serialization import jsonutils
from keystone.api._shared import EC2_S3_Resource
from keystone.api._shared import json_home_relations
from keystone.common import render_token
from keystone.common import utils
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class S3Api(ks_flask.APIBase):
    _name = 's3tokens'
    _import_name = __name__
    resources = []
    resource_mapping = [ks_flask.construct_resource_map(resource=S3Resource, url='/s3tokens', resource_kwargs={}, rel='s3tokens', resource_relation_func=json_home_relations.s3_token_resource_rel_func)]