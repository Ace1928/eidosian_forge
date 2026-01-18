from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class InstanceMysqlreplicaconfiguration(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'caCertificate': self.request.get('ca_certificate'), u'clientCertificate': self.request.get('client_certificate'), u'clientKey': self.request.get('client_key'), u'connectRetryInterval': self.request.get('connect_retry_interval'), u'dumpFilePath': self.request.get('dump_file_path'), u'masterHeartbeatPeriod': self.request.get('master_heartbeat_period'), u'password': self.request.get('password'), u'sslCipher': self.request.get('ssl_cipher'), u'username': self.request.get('username'), u'verifyServerCertificate': self.request.get('verify_server_certificate')})

    def from_response(self):
        return remove_nones_from_dict({u'caCertificate': self.request.get(u'caCertificate'), u'clientCertificate': self.request.get(u'clientCertificate'), u'clientKey': self.request.get(u'clientKey'), u'connectRetryInterval': self.request.get(u'connectRetryInterval'), u'dumpFilePath': self.request.get(u'dumpFilePath'), u'masterHeartbeatPeriod': self.request.get(u'masterHeartbeatPeriod'), u'password': self.request.get(u'password'), u'sslCipher': self.request.get(u'sslCipher'), u'username': self.request.get(u'username'), u'verifyServerCertificate': self.request.get(u'verifyServerCertificate')})