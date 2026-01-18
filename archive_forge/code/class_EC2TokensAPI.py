import urllib.parse
import flask
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_serialization import jsonutils
from keystone.api._shared import EC2_S3_Resource
from keystone.api._shared import json_home_relations
from keystone.common import render_token
from keystone.common import utils
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class EC2TokensAPI(ks_flask.APIBase):
    _name = 'ec2tokens'
    _import_name = __name__
    resources = []
    resource_mapping = [ks_flask.construct_resource_map(resource=EC2TokensResource, url='/ec2tokens', resource_kwargs={}, rel='ec2tokens', resource_relation_func=json_home_relations.os_ec2_resource_rel_func)]