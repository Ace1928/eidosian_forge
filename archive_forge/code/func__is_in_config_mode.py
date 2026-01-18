from __future__ import absolute_import, division, print_function
import getpass
import json
import logging
import os
import re
import signal
import socket
import time
import traceback
from functools import wraps
from io import BytesIO
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import cPickle
from ansible.playbook.play_context import PlayContext
from ansible.plugins.loader import cache_loader, cliconf_loader, connection_loader, terminal_loader
from ansible_collections.ansible.netcommon.plugins.connection.libssh import HAS_PYLIBSSH
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.connection_base import (
def _is_in_config_mode(self):
    """
        Check if the target device is in config mode by comparing
        the current prompt with the platform's `terminal_config_prompt`.
        Returns False if `terminal_config_prompt` is not defined.

        :returns: A boolean indicating if the device is in config mode or not.
        """
    cfg_mode = False
    cur_prompt = to_text(self.get_prompt(), errors='surrogate_then_replace').strip()
    cfg_prompt = getattr(self._terminal, 'terminal_config_prompt', None)
    if cfg_prompt and cfg_prompt.match(cur_prompt):
        cfg_mode = True
    return cfg_mode