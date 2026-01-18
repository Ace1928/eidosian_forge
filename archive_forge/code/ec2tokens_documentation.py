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
Authenticate ec2 token.

        POST /v3/ec2tokens
        