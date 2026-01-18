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
class UpdateActionWithAppend(UpdateAction):
    """Create a single dict value from delimited or repeated flags.

  This class provides a variant of UpdateAction, which allows for users to
  append, rather than reject, duplicate key values. For example, the user
  can specify:

    --inputs k1=v1a --inputs k1=v1b --inputs k2=v2

  and the result will be:

     { 'k1': ['v1a', 'v1b'], 'k2': 'v2' }
  """

    def OnDuplicateKeyAppend(self, key, existing_value=None, new_value=None):
        if existing_value is None:
            return key
        elif isinstance(existing_value, list):
            return existing_value + [new_value]
        else:
            return [existing_value, new_value]

    def __init__(self, option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None, onduplicatekey_handler=OnDuplicateKeyAppend):
        super(UpdateActionWithAppend, self).__init__(option_strings=option_strings, dest=dest, nargs=nargs, const=const, default=default, type=type, choices=choices, required=required, help=help, metavar=metavar, onduplicatekey_handler=onduplicatekey_handler)