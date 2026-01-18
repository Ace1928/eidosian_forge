from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.objects import AnsibleBaseYAMLObject
from ansible.playbook.delegatable import Delegatable
from ansible.playbook.role.definition import RoleDefinition
from ansible.module_utils.common.text.converters import to_native
class RoleInclude(RoleDefinition, Delegatable):
    """
    A derivative of RoleDefinition, used by playbook code when a role
    is included for execution in a play.
    """

    def __init__(self, play=None, role_basedir=None, variable_manager=None, loader=None, collection_list=None):
        super(RoleInclude, self).__init__(play=play, role_basedir=role_basedir, variable_manager=variable_manager, loader=loader, collection_list=collection_list)

    @staticmethod
    def load(data, play, current_role_path=None, parent_role=None, variable_manager=None, loader=None, collection_list=None):
        if not (isinstance(data, string_types) or isinstance(data, dict) or isinstance(data, AnsibleBaseYAMLObject)):
            raise AnsibleParserError('Invalid role definition: %s' % to_native(data))
        if isinstance(data, string_types) and ',' in data:
            raise AnsibleError('Invalid old style role requirement: %s' % data)
        ri = RoleInclude(play=play, role_basedir=current_role_path, variable_manager=variable_manager, loader=loader, collection_list=collection_list)
        return ri.load_data(data, variable_manager=variable_manager, loader=loader)