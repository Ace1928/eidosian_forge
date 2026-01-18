import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def _add_container_actions(self, container):
    title_group_map = {}
    for group in self._action_groups:
        if group.title in title_group_map:
            msg = _('cannot merge actions - two groups are named %r')
            raise ValueError(msg % group.title)
        title_group_map[group.title] = group
    group_map = {}
    for group in container._action_groups:
        if group.title not in title_group_map:
            title_group_map[group.title] = self.add_argument_group(title=group.title, description=group.description, conflict_handler=group.conflict_handler)
        for action in group._group_actions:
            group_map[action] = title_group_map[group.title]
    for group in container._mutually_exclusive_groups:
        mutex_group = self.add_mutually_exclusive_group(required=group.required)
        for action in group._group_actions:
            group_map[action] = mutex_group
    for action in container._actions:
        group_map.get(action, self)._add_action(action)