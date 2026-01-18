from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleParserError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.block import Block
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.helpers import load_list_of_blocks, load_list_of_roles
from ansible.playbook.role import Role, hash_params
from ansible.playbook.task import Task
from ansible.playbook.taggable import Taggable
from ansible.vars.manager import preprocess_vars
from ansible.utils.display import Display
@property
def ROLE_CACHE(self):
    """Backwards compat for custom strategies using ``play.ROLE_CACHE``
        """
    display.deprecated('Play.ROLE_CACHE is deprecated in favor of Play.role_cache, or StrategyBase._get_cached_role', version='2.18')
    cache = {}
    for path, roles in self.role_cache.items():
        for role in roles:
            name = role.get_name()
            hashed_params = hash_params(role._get_hash_dict())
            cache.setdefault(name, {})[hashed_params] = role
    return cache