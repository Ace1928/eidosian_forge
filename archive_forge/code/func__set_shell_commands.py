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
def _set_shell_commands(self, cmds_dict):
    for k, v in cmds_dict.items():
        self.command_manager.add_command(k, v)