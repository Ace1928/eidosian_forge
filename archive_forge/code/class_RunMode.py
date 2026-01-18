from __future__ import absolute_import, division, print_function
import copy
import glob
import os
from importlib import import_module
from ansible.errors import AnsibleActionFail, AnsibleError
from ansible.module_utils._text import to_text
from ansible.utils.display import Display
from ansible_collections.ansible.netcommon.plugins.action.network import (
class RunMode:
    RM_LIST = 0
    RM_GET = 1
    RM_CONFIG = 1