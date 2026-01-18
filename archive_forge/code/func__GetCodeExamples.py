import argparse
import arg_parsers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import decimal
import json
import re
from dateutil import tz
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import zip  # pylint: disable=redefined-builtin
def _GetCodeExamples(self, flag_name):
    """Returns a string of user input examples."""
    shorthand_example = usage_text.FormatCodeSnippet(arg_name=flag_name, arg_value=self.GetUsageExample(shorthand=True), append=self.repeated)
    json_example = usage_text.FormatCodeSnippet(arg_name=flag_name, arg_value=self.GetUsageExample(shorthand=False))
    file_example = usage_text.FormatCodeSnippet(arg_name=flag_name, arg_value='path_to_file.(yaml|json)')
    if shorthand_example == json_example:
        return '*Input Example:*\n\n{}\n\n*File Example:*\n\n{}'.format(shorthand_example, file_example)
    else:
        return '*Shorthand Example:*\n\n{}\n\n*JSON Example:*\n\n{}\n\n*File Example:*\n\n{}'.format(shorthand_example, json_example, file_example)