import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
def _find_actions(self, subparsers, actions_module):
    for attr in (a for a in dir(actions_module) if a.startswith('do_')):
        command = attr[3:].replace('_', '-')
        callback = getattr(actions_module, attr)
        desc = callback.__doc__ or ''
        action_help = desc.strip()
        arguments = getattr(callback, 'arguments', [])
        group_args = getattr(callback, 'deprecated_groups', [])
        subparser = subparsers.add_parser(command, help=action_help, description=desc, add_help=False, formatter_class=OpenStackHelpFormatter)
        subparser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)
        self.subcommands[command] = subparser
        for old_info, new_info, req in group_args:
            group = subparser.add_mutually_exclusive_group(required=req)
            group.add_argument(*old_info[0], **old_info[1])
            group.add_argument(*new_info[0], **new_info[1])
        for args, kwargs in arguments:
            subparser.add_argument(*args, **kwargs)
        subparser.set_defaults(func=callback)