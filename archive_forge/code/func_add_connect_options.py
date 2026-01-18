from __future__ import (absolute_import, division, print_function)
import copy
import operator
import argparse
import os
import os.path
import sys
import time
from jinja2 import __version__ as j2_version
import ansible
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.yaml import HAS_LIBYAML, yaml_load
from ansible.release import __version__
from ansible.utils.path import unfrackpath
def add_connect_options(parser):
    """Add options for commands which need to connection to other hosts"""
    connect_group = parser.add_argument_group('Connection Options', 'control as whom and how to connect to hosts')
    connect_group.add_argument('--private-key', '--key-file', default=C.DEFAULT_PRIVATE_KEY_FILE, dest='private_key_file', help='use this file to authenticate the connection', type=unfrack_path())
    connect_group.add_argument('-u', '--user', default=C.DEFAULT_REMOTE_USER, dest='remote_user', help='connect as this user (default=%s)' % C.DEFAULT_REMOTE_USER)
    connect_group.add_argument('-c', '--connection', dest='connection', default=C.DEFAULT_TRANSPORT, help='connection type to use (default=%s)' % C.DEFAULT_TRANSPORT)
    connect_group.add_argument('-T', '--timeout', default=None, type=int, dest='timeout', help='override the connection timeout in seconds (default depends on connection)')
    connect_group.add_argument('--ssh-common-args', default=None, dest='ssh_common_args', help='specify common arguments to pass to sftp/scp/ssh (e.g. ProxyCommand)')
    connect_group.add_argument('--sftp-extra-args', default=None, dest='sftp_extra_args', help='specify extra arguments to pass to sftp only (e.g. -f, -l)')
    connect_group.add_argument('--scp-extra-args', default=None, dest='scp_extra_args', help='specify extra arguments to pass to scp only (e.g. -l)')
    connect_group.add_argument('--ssh-extra-args', default=None, dest='ssh_extra_args', help='specify extra arguments to pass to ssh only (e.g. -R)')
    parser.add_argument_group(connect_group)
    connect_password_group = parser.add_mutually_exclusive_group()
    connect_password_group.add_argument('-k', '--ask-pass', default=C.DEFAULT_ASK_PASS, dest='ask_pass', action='store_true', help='ask for connection password')
    connect_password_group.add_argument('--connection-password-file', '--conn-pass-file', default=C.CONNECTION_PASSWORD_FILE, dest='connection_password_file', help='Connection password file', type=unfrack_path(), action='store')
    parser.add_argument_group(connect_password_group)