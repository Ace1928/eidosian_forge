from __future__ import (absolute_import, division, print_function)
import json
import ssl
from time import sleep
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _host_group_name_for_uuid(self, hosts, host_uuid):
    for host in hosts:
        if host == host_uuid:
            return 'xo_host_{0}'.format(clean_group_name(hosts[host_uuid]['name_label']))