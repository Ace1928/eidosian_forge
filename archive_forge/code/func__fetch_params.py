from __future__ import (absolute_import, division, print_function)
import copy
import json
from ansible_collections.theforeman.foreman.plugins.module_utils._version import LooseVersion
from time import sleep
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, to_safe_group_name, Constructable
def _fetch_params(self):
    options = ('no', 'yes')
    params = dict()
    report_options = self.get_option('report') or {}
    self.want_location = report_options.get('want_location', self.get_option('want_location'))
    self.want_organization = report_options.get('want_organization', self.get_option('want_organization'))
    self.want_IPv4 = report_options.get('want_ipv4', self.get_option('want_ipv4'))
    self.want_IPv6 = report_options.get('want_ipv6', self.get_option('want_ipv6'))
    self.want_host_group = report_options.get('want_host_group', self.get_option('want_host_group'))
    self.want_hostcollections = report_options.get('want_hostcollections', self.get_option('want_hostcollections'))
    self.want_subnet = report_options.get('want_subnet', self.get_option('want_subnet'))
    self.want_subnet_v6 = report_options.get('want_subnet_v6', self.get_option('want_subnet_v6'))
    self.want_smart_proxies = report_options.get('want_smart_proxies', self.get_option('want_smart_proxies'))
    self.want_content_facet_attributes = report_options.get('want_content_facet_attributes', self.get_option('want_content_facet_attributes'))
    self.want_params = self.get_option('want_params')
    self.want_facts = self.get_option('want_facts')
    self.host_filters = self.get_option('host_filters')
    params['Organization'] = options[self.want_organization]
    params['Location'] = options[self.want_location]
    params['IPv4'] = options[self.want_IPv4]
    params['IPv6'] = options[self.want_IPv6]
    params['Facts'] = options[self.want_facts]
    params['Host Group'] = options[self.want_host_group]
    params['Host Collections'] = options[self.want_hostcollections]
    params['Subnet'] = options[self.want_subnet]
    params['Subnet v6'] = options[self.want_subnet_v6]
    params['Smart Proxies'] = options[self.want_smart_proxies]
    params['Content Attributes'] = options[self.want_content_facet_attributes]
    params['Host Parameters'] = options[self.want_params]
    if self.host_filters:
        params['Hosts'] = self.host_filters
    return params