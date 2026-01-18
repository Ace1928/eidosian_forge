import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class VCloudConnection(ConnectionUserAndKey):
    """
    Connection class for the vCloud driver
    """
    responseCls = VCloudResponse
    token = None
    host = None

    def request(self, *args, **kwargs):
        self._get_auth_token()
        return super().request(*args, **kwargs)

    def check_org(self):
        self._get_auth_token()

    def _get_auth_headers(self):
        """Some providers need different headers than others"""
        return {'Authorization': 'Basic %s' % base64.b64encode(b('{}:{}'.format(self.user_id, self.key))).decode('utf-8'), 'Content-Length': '0', 'Accept': 'application/*+xml'}

    def _get_auth_token(self):
        if not self.token:
            self.connection.request(method='POST', url='/api/v0.8/login', headers=self._get_auth_headers())
            resp = self.connection.getresponse()
            headers = resp.headers
            body = ET.XML(resp.text)
            try:
                self.token = headers['set-cookie']
            except KeyError:
                raise InvalidCredsError()
            self.driver.org = get_url_path(body.find(fixxpath(body, 'Org')).get('href'))

    def add_default_headers(self, headers):
        headers['Cookie'] = self.token
        headers['Accept'] = 'application/*+xml'
        return headers