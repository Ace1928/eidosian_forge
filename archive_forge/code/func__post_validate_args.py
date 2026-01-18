from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.parsing.mod_args import ModuleArgsParser
from ansible.parsing.yaml.objects import AnsibleBaseYAMLObject, AnsibleMapping
from ansible.plugins.loader import lookup_loader
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.block import Block
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.conditional import Conditional
from ansible.playbook.delegatable import Delegatable
from ansible.playbook.loop_control import LoopControl
from ansible.playbook.notifiable import Notifiable
from ansible.playbook.role import Role
from ansible.playbook.taggable import Taggable
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
def _post_validate_args(self, attr, value, templar):
    self.untemplated_args = value
    args = templar.template(value)
    if '_variable_params' in args:
        variable_params = args.pop('_variable_params')
        if isinstance(variable_params, dict):
            if C.INJECT_FACTS_AS_VARS:
                display.warning("Using a variable for a task's 'args' is unsafe in some situations (see https://docs.ansible.com/ansible/devel/reference_appendices/faq.html#argsplat-unsafe)")
            variable_params.update(args)
            args = variable_params
        else:
            raise AnsibleError(f"invalid or malformed argument: '{variable_params}'")
    return args