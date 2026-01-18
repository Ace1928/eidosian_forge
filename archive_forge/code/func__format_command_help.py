import argparse
import collections
import datetime
import functools
import os
import sys
import time
import uuid
from oslo_utils import encodeutils
import prettytable
from glance.common import exception
import glance.image_cache.client
from glance.version import version_info as version
def _format_command_help():
    """Formats the help string for subcommands."""
    help_msg = 'Commands:\n\n'
    for command, info in CACHE_COMMANDS.items():
        if command == 'help':
            command = 'help <command>'
        help_msg += '    %-28s%s\n\n' % (command, info[1])
    return help_msg