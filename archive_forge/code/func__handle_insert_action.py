from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _handle_insert_action(self, action, item):
    """Handle the nuances of the insert type

        :param action:
        :param item:
        :return:
        """
    action['type'] = 'insert'
    options = ['http_header', 'http_referer', 'http_set_cookie', 'http_cookie']
    if not any((x for x in options if x in item)):
        raise F5ModuleError("A 'http_header', 'http_referer', 'http_set_cookie' or 'http_cookie' must be specified when the 'insert' type is used.")
    if 'http_header' in item and item['http_header']:
        if item['http_header']['value'] is None:
            raise F5ModuleError("The http_header value key is required when action is of type 'insert'.")
        if item['http_header']['event'] == 'request':
            action.update(httpHeader=True, tmName=item['http_header']['name'], value=item['http_header']['value'], request=True)
        elif item['http_header']['event'] == 'response':
            action.update(httpHeader=True, tmName=item['http_header']['name'], value=item['http_header']['value'], response=True)
        else:
            action.update(httpHeader=True, tmName=item['http_header']['name'], value=item['http_header']['value'])
    if 'http_referer' in item and item['http_referer']:
        if item['http_referer']['value'] is None:
            raise F5ModuleError("The http_referer value key is required when action is of type 'insert'.")
        if item['http_referer']['event'] == 'request':
            action.update(httpReferer=True, value=item['http_referer']['value'], request=True)
        if item['http_referer']['event'] == 'proxy_connect':
            action.update(httpReferer=True, value=item['http_referer']['value'], proxyConnect=True)
        if item['http_referer']['event'] == 'proxy_request':
            action.update(httpReferer=True, value=item['http_referer']['value'], proxyRequest=True)
    if 'http_cookie' in item and item['http_cookie']:
        if item['http_cookie']['value'] is None:
            raise F5ModuleError("The http_cookie value key is required when action is of type 'insert'.")
        if item['http_cookie']['event'] == 'request':
            action.update(httpCookie=True, tmName=item['http_cookie']['name'], value=item['http_cookie']['value'], request=True)
        elif item['http_cookie']['event'] == 'proxy_connect':
            action.update(httpCookie=True, tmName=item['http_cookie']['name'], value=item['http_cookie']['value'], proxyConnect=True)
        elif item['http_cookie']['event'] == 'proxy_request':
            action.update(httpCookie=True, tmName=item['http_cookie']['name'], value=item['http_cookie']['value'], proxyRequest=True)
        else:
            action.update(httpCookie=True, tmName=item['http_cookie']['name'], value=item['http_cookie']['value'])
    if 'http_set_cookie' in item and item['http_set_cookie']:
        if item['http_set_cookie']['value'] is None:
            raise F5ModuleError("The http_set_cookie value key is required when action is of type 'insert'.")
        action.update(httpSetCookie=True, tmName=item['http_set_cookie']['name'], value=item['http_set_cookie']['value'], response=True)