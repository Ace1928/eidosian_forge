import argparse
import logging
import os
import sys
from cliff import app
from cliff import commandmanager
from osc_lib.command import command
from mistralclient.api import client
from mistralclient.auth import auth_types
import mistralclient.commands.v2.action_executions
import mistralclient.commands.v2.actions
import mistralclient.commands.v2.code_sources
import mistralclient.commands.v2.cron_triggers
import mistralclient.commands.v2.dynamic_actions
import mistralclient.commands.v2.environments
import mistralclient.commands.v2.event_triggers
import mistralclient.commands.v2.executions
import mistralclient.commands.v2.members
import mistralclient.commands.v2.services
import mistralclient.commands.v2.tasks
import mistralclient.commands.v2.workbooks
import mistralclient.commands.v2.workflows
from mistralclient import exceptions as exe
class BashCompletionCommand(command.Command):
    """Prints all of the commands and options for bash-completion."""

    def take_action(self, parsed_args):
        commands = set()
        options = set()
        for option, _action in self.app.parser._option_string_actions.items():
            options.add(option)
        for command_name, _cmd in self.app.command_manager:
            commands.add(command_name)
        print(' '.join(commands | options))