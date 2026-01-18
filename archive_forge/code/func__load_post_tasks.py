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
def _load_post_tasks(self, attr, ds):
    """
        Loads a list of blocks from a list which may be mixed tasks/blocks.
        Bare tasks outside of a block are given an implicit block.
        """
    try:
        return load_list_of_blocks(ds=ds, play=self, variable_manager=self._variable_manager, loader=self._loader)
    except AssertionError as e:
        raise AnsibleParserError('A malformed block was encountered while loading post_tasks', obj=self._ds, orig_exc=e)