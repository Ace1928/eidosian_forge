from __future__ import (absolute_import, division, print_function)
from ansible.module_utils import basic
class AzureRMModuleBaseMock:
    """ Mock for sanity tests when azcollection is not installed """

    def __init__(self, derived_arg_spec, required_if=None, supports_check_mode=False, supports_tags=True, **kwargs):
        if supports_tags:
            derived_arg_spec.update(dict(tags=dict()))
        self.module = basic.AnsibleModule(argument_spec=derived_arg_spec, required_if=required_if, supports_check_mode=supports_check_mode)
        self.module.warn('Running in Unit Test context!')
        self.parameters = dict([item for item in self.module.params.items() if item[1] is not None])
        self.module_arg_spec = dict([item for item in self.module_arg_spec.items() if item[0] in self.parameters])

    def update_tags(self, tags):
        self.module.log('update_tags called with:', tags)
        return (None, None)