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
class HelpAction(argparse.Action):
    """Custom help action.

    Provide a custom action so the -h and --help options
    to the main app will print a list of the commands.

    The commands are determined by checking the CommandManager
    instance, passed in as the "default" value for the action.

    """

    def __call__(self, parser, namespace, values, option_string=None):
        outputs = []
        max_len = 0
        app = self.default
        parser.print_help(app.stdout)
        app.stdout.write('\nCommands for API v2 :\n')
        for name, ep in sorted(app.command_manager):
            factory = ep.load()
            cmd = factory(self, None)
            one_liner = cmd.get_description().split('\n')[0]
            outputs.append((name, one_liner))
            max_len = max(len(name), max_len)
        for name, one_liner in outputs:
            app.stdout.write('  %s  %s\n' % (name.ljust(max_len), one_liner))
        sys.exit(0)