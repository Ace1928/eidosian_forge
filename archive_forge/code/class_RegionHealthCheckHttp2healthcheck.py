from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class RegionHealthCheckHttp2healthcheck(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'host': self.request.get('host'), u'requestPath': self.request.get('request_path'), u'response': self.request.get('response'), u'port': self.request.get('port'), u'portName': self.request.get('port_name'), u'proxyHeader': self.request.get('proxy_header'), u'portSpecification': self.request.get('port_specification')})

    def from_response(self):
        return remove_nones_from_dict({u'host': self.request.get(u'host'), u'requestPath': self.request.get(u'requestPath'), u'response': self.request.get(u'response'), u'port': self.request.get(u'port'), u'portName': self.request.get(u'portName'), u'proxyHeader': self.request.get(u'proxyHeader'), u'portSpecification': self.request.get(u'portSpecification')})