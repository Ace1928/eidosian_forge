from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.mh.base import ModuleHelperBase, AnsibleModule  # noqa: F401
from ansible_collections.community.general.plugins.module_utils.mh.mixins.state import StateMixin
from ansible_collections.community.general.plugins.module_utils.mh.mixins.deps import DependencyMixin
from ansible_collections.community.general.plugins.module_utils.mh.mixins.vars import VarsMixin
from ansible_collections.community.general.plugins.module_utils.mh.mixins.deprecate_attrs import DeprecateAttrsMixin
def _vars_changed(self):
    return any((self.vars.has_changed(v) for v in self.vars.change_vars()))