from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.argspec.acls.acls import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.config.acls.acls import Acls

    Main entry point for module execution
    :returns: the result form module invocation
    