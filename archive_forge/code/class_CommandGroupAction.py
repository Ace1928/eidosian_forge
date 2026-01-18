from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import argparse
import collections
import io
import itertools
import os
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base  # pylint: disable=unused-import
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import suggest_commands
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
import six
class CommandGroupAction(CloudSDKSubParsersAction):
    """A subparser for loading calliope command groups on demand.

  We use this to intercept the parsing right before it needs to start parsing
  args for sub groups and we then load the specific sub group it needs.
  """

    def __init__(self, *args, **kwargs):
        self._calliope_command = kwargs.pop('calliope_command')
        super(CommandGroupAction, self).__init__(*args, **kwargs)

    def IsValidChoice(self, choice):
        if '_ARGCOMPLETE' in os.environ:
            self._calliope_command.LoadSubElement(choice)
        return self._calliope_command.IsValidSubElement(choice)

    def LoadAllChoices(self):
        self._calliope_command.LoadAllSubElements()

    def __call__(self, parser, namespace, values, option_string=None):
        parser_name = values[0]
        if self._calliope_command:
            self._calliope_command.LoadSubElement(parser_name)
        super(CommandGroupAction, self).__call__(parser, namespace, values, option_string=option_string)