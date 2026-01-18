from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common import get_connect_params
from ansible_collections.community.docker.plugins.module_utils.util import update_tls_hostname
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.parsing.utils.addresses import parse_address
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.library_inventory_filtering_v1.plugins.plugin_utils.inventory_filter import parse_filters, filter_host
Return the possibly of a file being consumable by this plugin.