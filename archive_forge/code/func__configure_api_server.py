import http.client as http
import os
import socket
import time
from oslo_serialization import jsonutils
from oslo_utils.fixture import uuidsentinel as uuids
import requests
from glance.common import wsgi
from glance.tests import functional
def _configure_api_server(self):
    self.my_api_server.deployment_flavor = 'noauth'