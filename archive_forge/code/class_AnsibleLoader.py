from __future__ import (absolute_import, division, print_function)
from yaml.resolver import Resolver
from ansible.parsing.yaml.constructor import AnsibleConstructor
from ansible.module_utils.common.yaml import HAS_LIBYAML, Parser
class AnsibleLoader(Reader, Scanner, Parser, Composer, AnsibleConstructor, Resolver):

    def __init__(self, stream, file_name=None, vault_secrets=None):
        Reader.__init__(self, stream)
        Scanner.__init__(self)
        Parser.__init__(self)
        Composer.__init__(self)
        AnsibleConstructor.__init__(self, file_name=file_name, vault_secrets=vault_secrets)
        Resolver.__init__(self)