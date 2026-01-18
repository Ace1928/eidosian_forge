import argparse
import fnmatch
import importlib
import inspect
import re
import sys
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils import statemachine
from cliff import app
from cliff import commandmanager
def _generate_command_nodes(self, commands, application_name):
    ignored_opts = self._get_ignored_opts()
    output = []
    for command_name in sorted(commands):
        command_class = commands[command_name]
        title = command_name
        if application_name:
            command_name = ' '.join([application_name, command_name])
        output.extend(self._generate_nodes_per_command(title, command_name, command_class, ignored_opts))
    return output