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
def _RejoinJsonStrs(json_list, delim, arg_value):
    """Rejoins json substrings that are part of the same json strings.

  For example:
      [
          'key={"a":"b"',
          '"c":"d"}'
      ]

  Is merged together into: ['key={"a":"b","c":"d"}']

  Args:
    json_list: [str], list of json snippets
    delim: str, delim used to rejoin the json snippets
    arg_value: str, original value used to make json_list

  Returns:
    list of strings containing balanced json
  """
    result = []
    current_substr = None
    for token in json_list:
        if not current_substr:
            current_substr = token
        else:
            current_substr += delim + token
        if _ContainsValidJson(current_substr):
            result.append(current_substr)
            current_substr = None
    if current_substr:
        raise ValueError('Invalid entry "{}": missing opening brace ("{{" or "[") or closing brace ("}}" or "]").'.format(arg_value))
    return result