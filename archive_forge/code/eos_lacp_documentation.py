from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.arista.eos.plugins.module_utils.network.eos.argspec.lacp.lacp import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.config.lacp.lacp import Lacp

    Main entry point for module execution

    :returns: the result form module invocation
    