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