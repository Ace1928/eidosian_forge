from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import copy
import json
import shlex
from googlecloudsdk import gcloud_main
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _validate_command_prefix(self, command_arguments, command_string):
    """Validate that the argument string contains a valid command or group."""
    cli = gcloud_main.CreateCLI([])
    command_arguments = command_arguments[1:]
    index = 0
    current_command_node = cli._TopElement()
    for argument in command_arguments:
        if argument.startswith('--'):
            return (True, current_command_node, command_arguments[index:])
        current_command_node = current_command_node.LoadSubElement(argument)
        if not current_command_node:
            self._store_validation_results(False, command_string, "Invalid choice: '{}'".format(argument), 'UnrecognizedCommandError')
            return (False, None, None)
        index += 1
        if not current_command_node.is_group:
            return (True, current_command_node, command_arguments[index:])
    remaining_flags = command_arguments[index:]
    if not remaining_flags:
        self._store_validation_results(False, command_string, 'Command name argument expected', 'UnrecognizedCommandError')
        return (False, None, None)
    raise CommandValidationError('Command could not be validated due to unforeseen edge case.')