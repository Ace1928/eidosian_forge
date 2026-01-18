import flask
from flask import request
import http.client
from oslo_serialization import jsonutils
from keystone.common import json_home
import keystone.conf
from keystone.server import flask as ks_flask
class DiscoveryAPI(object):

    @staticmethod
    def instantiate_and_register_to_app(flask_app):
        flask_app.register_blueprint(_DISCOVERY_BLUEPRINT)