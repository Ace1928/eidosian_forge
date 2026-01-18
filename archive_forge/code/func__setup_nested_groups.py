from __future__ import absolute_import, division, print_function
import json
import uuid
import math
import os
import datetime
from copy import deepcopy
from functools import partial
from sys import version as python_version
from threading import Thread
from typing import Iterable
from itertools import chain
from collections import defaultdict
from ipaddress import ip_interface
from ansible.constants import DEFAULT_LOCAL_TMP
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six import raise_from
def _setup_nested_groups(self, group, lookup, parent_lookup):
    transformed_group_names = dict()
    for obj_id in lookup:
        group_name = self.generate_group_name(group, lookup[obj_id])
        transformed_group_names[obj_id] = self.inventory.add_group(group=group_name)
    for obj_id in lookup:
        group_name = transformed_group_names[obj_id]
        parent_id = parent_lookup.get(obj_id, None)
        if parent_id is not None and parent_id in transformed_group_names:
            parent_name = transformed_group_names[parent_id]
            self.inventory.add_child(parent_name, group_name)
    return transformed_group_names