from __future__ import (absolute_import, division, print_function)
import ansible.constants as C
from ansible.errors import AnsibleParserError
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.conditional import Conditional
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.delegatable import Delegatable
from ansible.playbook.helpers import load_list_of_tasks
from ansible.playbook.notifiable import Notifiable
from ansible.playbook.role import Role
from ansible.playbook.taggable import Taggable
from ansible.utils.sentinel import Sentinel
def _load_block(self, attr, ds):
    try:
        return load_list_of_tasks(ds, play=self._play, block=self, role=self._role, task_include=None, variable_manager=self._variable_manager, loader=self._loader, use_handlers=self._use_handlers)
    except AssertionError as e:
        raise AnsibleParserError('A malformed block was encountered while loading a block', obj=self._ds, orig_exc=e)