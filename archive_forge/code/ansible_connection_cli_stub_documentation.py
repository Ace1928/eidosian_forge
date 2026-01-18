from __future__ import (absolute_import, division, print_function)
import fcntl
import hashlib
import io
import os
import pickle
import signal
import socket
import sys
import time
import traceback
import errno
import json
from contextlib import contextmanager
from ansible import constants as C
from ansible.cli.arguments import option_helpers as opt_help
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError, send_data, recv_data
from ansible.module_utils.service import fork_process
from ansible.parsing.ajson import AnsibleJSONEncoder, AnsibleJSONDecoder
from ansible.playbook.play_context import PlayContext
from ansible.plugins.loader import connection_loader, init_plugin_loader
from ansible.utils.path import unfrackpath, makedirs_safe
from ansible.utils.display import Display
from ansible.utils.jsonrpc import JsonRpcServer
 Shuts down the local domain socket
        