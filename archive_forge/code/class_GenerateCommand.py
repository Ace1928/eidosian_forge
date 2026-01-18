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
class GenerateCommand(base.Command):
    """Generate YAML file to implement given command.

  The command YAML file is generated in the --output-dir directory.
  """
    _VALIDATION_RESULTS = []

    def _validate_command(self, command_string):
        """Validate a single command."""
        command_arguments = _separate_command_arguments(command_string)
        command_success, command_node, flag_arguments = self._validate_command_prefix(command_arguments, command_string)
        if not command_success:
            return
        flag_success = self._validate_command_suffix(command_node, flag_arguments, command_string)
        if not flag_success:
            return
        self._store_validation_results(True, command_string)

    def _validate_commands_from_file(self, commands_file):
        """Validate multiple commands given in a file."""
        commands = _read_commands_from_file(commands_file)
        for command in commands:
            self._validate_command(command)

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

    def _validate_command_suffix(self, command_node, command_arguments, command_string):
        """Validates that the given flags can be parsed by the argparse parser."""
        found_parent = False
        if command_arguments:
            for command_arg in command_arguments:
                if '--project' in command_arg or '--folder' in command_arg or '--organization' in command_arg:
                    found_parent = True
        if not command_arguments:
            command_arguments = []
        if not found_parent:
            command_arguments.append('--project=myproject')
        try:
            command_node._parser.parse_args(command_arguments, raise_error=True)
        except argparse.ArgumentError as e:
            self._store_validation_results(False, command_string, six.text_type(e), type(e).__name__)
            return False
        return True

    def _store_validation_results(self, success, command_string, error_message=None, error_type=None):
        """Store information related to the command validation."""
        validation_output = copy.deepcopy(_PARSING_OUTPUT_TEMPLATE)
        validation_output['command_string'] = command_string
        validation_output['success'] = success
        validation_output['error_message'] = error_message
        validation_output['error_type'] = error_type
        self._VALIDATION_RESULTS.append(validation_output)

    def _log_validation_results(self):
        """Output collected validation results."""
        log.out.Print(json.dumps(self._VALIDATION_RESULTS))

    @staticmethod
    def Args(parser):
        command_group = parser.add_group(mutex=True)
        command_group.add_argument('--command-string', help='Gcloud command to statically validate.')
        command_group.add_argument('--commands-file', help='JSON file containing list of gcloud commands to validate.')

    def Run(self, args):
        if args.IsSpecified('command_string'):
            self._validate_command(args.command_string)
        else:
            self._validate_commands_from_file(args.commands_file)
        self._log_validation_results()