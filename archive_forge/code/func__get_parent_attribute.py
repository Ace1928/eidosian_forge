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
def _get_parent_attribute(self, attr, omit=False):
    """
        Generic logic to get the attribute or parent attribute for a task value.
        """
    fattr = self.fattributes[attr]
    extend = fattr.extend
    prepend = fattr.prepend
    try:
        if omit:
            value = Sentinel
        else:
            value = getattr(self, f'_{attr}', Sentinel)
        if getattr(self._parent, 'statically_loaded', True):
            _parent = self._parent
        else:
            _parent = self._parent._parent
        if _parent and (value is Sentinel or extend):
            if getattr(_parent, 'statically_loaded', True):
                if attr != 'vars' and hasattr(_parent, '_get_parent_attribute'):
                    parent_value = _parent._get_parent_attribute(attr)
                else:
                    parent_value = getattr(_parent, f'_{attr}', Sentinel)
                if extend:
                    value = self._extend_value(value, parent_value, prepend)
                else:
                    value = parent_value
    except KeyError:
        pass
    return value