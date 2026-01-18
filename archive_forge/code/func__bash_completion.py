import argparse
import logging
import os
import sys
from cliff import app
from cliff import commandmanager
from keystoneauth1 import loading
from oslo_utils import encodeutils
from blazarclient import client as blazar_client
from blazarclient import exception
from blazarclient.v1.shell_commands import allocations
from blazarclient.v1.shell_commands import floatingips
from blazarclient.v1.shell_commands import hosts
from blazarclient.v1.shell_commands import leases
from blazarclient import version as base_version
def _bash_completion(self):
    """Prints all of the commands and options for bash-completion."""
    commands = set()
    options = set()
    for option, _action in self.parser._option_string_actions.items():
        options.add(option)
    for command_name, command in self.command_manager:
        commands.add(command_name)
        cmd_factory = command.load()
        cmd = cmd_factory(self, None)
        cmd_parser = cmd.get_parser('')
        for option, _action in cmd_parser._option_string_actions.items():
            options.add(option)
    print(' '.join(commands | options))