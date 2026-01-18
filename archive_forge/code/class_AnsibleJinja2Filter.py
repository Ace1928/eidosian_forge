from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible.plugins import AnsibleJinja2Plugin
class AnsibleJinja2Filter(AnsibleJinja2Plugin):

    def _no_options(self, *args, **kwargs):
        raise NotImplementedError('Jinja2 filter plugins do not support option functions, they use direct arguments instead.')