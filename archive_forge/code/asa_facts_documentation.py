from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.argspec.facts.facts import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.asa import asa_argument_spec
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts

    Main entry point for module execution

    :returns: ansible_facts
    