from __future__ import (absolute_import, division, print_function)
import json
from ansible.errors import AnsibleParserError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _query_hosts(self, hosts=None, attrs=None, joins=None, host_filter=None):
    query_hosts_url = '{0}/objects/hosts'.format(self.icinga2_url)
    self.headers['X-HTTP-Method-Override'] = 'GET'
    data_dict = dict()
    if hosts:
        data_dict['hosts'] = hosts
    if attrs is not None:
        data_dict['attrs'] = attrs
    if joins is not None:
        data_dict['joins'] = joins
    if host_filter is not None:
        data_dict['filter'] = host_filter.replace('\\"', '"')
        self.display.vvv(host_filter)
    host_dict = self._post_request(query_hosts_url, data_dict)
    return host_dict['results']