import logging
import time
import requests
from oslo_utils import importutils
from troveclient.apiclient import exceptions
def _http_log_req(self, method, url, kwargs):
    if not self.debug:
        return
    string_parts = ['curl -i', "-X '%s'" % method, "'%s'" % url]
    for element in kwargs['headers']:
        header = "-H '%s: %s'" % (element, kwargs['headers'][element])
        string_parts.append(header)
    LOG.debug('REQ: %s', ' '.join(string_parts))
    if 'data' in kwargs:
        LOG.debug('REQ BODY: %s\n', kwargs['data'])