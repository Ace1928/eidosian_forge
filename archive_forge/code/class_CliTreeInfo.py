from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
class CliTreeInfo(object):
    """Info for one CLI tree. A list of these is returned by ListAll()."""

    def __init__(self, command, path, version, cli_version, command_installed, error):
        self.command = command
        self.path = path
        self.version = version
        self.cli_version = cli_version
        self.command_installed = command_installed
        self.error = error