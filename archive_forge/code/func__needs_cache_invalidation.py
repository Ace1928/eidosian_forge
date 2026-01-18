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
def _needs_cache_invalidation(self, command):
    """
        This method determines if it is necessary to invalidate
        the existing cache based on whether the device has entered
        configuration mode or if the last command sent to the device
        is potentially capable of making configuration changes.

        :param command: The last command sent to the target device.
        :returns: A boolean indicating if cache invalidation is required or not.
        """
    invalidate = False
    cfg_cmds = []
    try:
        cfg_cmds = self.cliconf.get_option('config_commands')
    except AttributeError:
        cfg_cmds = []
    if self._is_in_config_mode() or to_text(command) in cfg_cmds:
        invalidate = True
    return invalidate