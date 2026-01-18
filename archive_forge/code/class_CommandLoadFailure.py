from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import importlib
import os
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_release_tracks
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import pkg_resources
from ruamel import yaml
import six
class CommandLoadFailure(Exception):
    """An exception for when a command or group module cannot be imported."""

    def __init__(self, command, root_exception):
        self.command = command
        self.root_exception = root_exception
        super(CommandLoadFailure, self).__init__('Problem loading {command}: {issue}.'.format(command=command, issue=six.text_type(root_exception)))