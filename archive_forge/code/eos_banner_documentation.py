from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.arista.eos.plugins.module_utils.network.eos.eos import (
main entry point for module execution