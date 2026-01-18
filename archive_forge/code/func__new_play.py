from __future__ import (absolute_import, division, print_function)
import datetime
import json
import copy
from functools import partial
from ansible.inventory.host import Host
from ansible.module_utils._text import to_text
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.plugins.callback import CallbackBase
def _new_play(self, play):
    self._is_lockstep = play.strategy in LOCKSTEP_CALLBACKS
    return {'play': {'name': play.get_name(), 'id': to_text(play._uuid), 'path': to_text(play.get_path()), 'duration': {'start': current_time()}}, 'tasks': []}