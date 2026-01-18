from __future__ import (absolute_import, division, print_function)
from ansible.plugins import AnsibleJinja2Plugin
class AnsibleJinja2Test(AnsibleJinja2Plugin):

    def _no_options(self, *args, **kwargs):
        raise NotImplementedError('Jinaj2 test plugins do not support option functions, they use direct arguments instead.')