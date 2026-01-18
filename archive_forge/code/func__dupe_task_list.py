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
def _dupe_task_list(task_list, new_block):
    new_task_list = []
    for task in task_list:
        new_task = task.copy(exclude_parent=True)
        if task._parent:
            new_task._parent = task._parent.copy(exclude_tasks=True)
            if task._parent == new_block:
                new_task._parent = new_block
            else:
                cur_obj = new_task._parent
                while cur_obj._parent and cur_obj._parent != new_block:
                    cur_obj = cur_obj._parent
                cur_obj._parent = new_block
        else:
            new_task._parent = new_block
        new_task_list.append(new_task)
    return new_task_list