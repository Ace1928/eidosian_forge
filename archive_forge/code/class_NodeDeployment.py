from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.personas_utils import Node
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
class NodeDeployment(object):

    def requires_update(self, current_obj, requested_obj):
        obj_params = [('roles', 'roles'), ('services', 'services')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))