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
def FlattenAction(dedup=True):
    """Creates an action that concats flag values.

  Args:
    dedup: bool, determines whether values in generated list should be unique

  Returns:
    A custom argparse action that flattens the values before adding
    to namespace
  """

    class Action(argparse.Action):
        """Create a single list from delimited flags.

    For example, with the following flag defition:

        parser.add_argument(
            '--inputs',
            type=arg_parsers.ArgObject(repeated=True),
            action=FlattenAction(dedup=True))

    a caller can specify on the command line flags such as:

        --inputs '["v1", "v2"]' --inputs '["v3", "v4"]'

    and the result will be one list of non-repeating values:

        ["v1", "v2", "v3", "v4"]

    Recommend using this action with ArgObject where `repeated` is set to True.
    This allows users to set the list either with append action or with
    one json list. For example, all below examples result
    in ["v1", "v2", "v3", "v4"]

        1) --inputs v1 --inputs v2 --inputs v3 --inputs v4
        2) --inputs '["v1", "v2", "v3", "v4"]'
    """

        def __call__(self, parser, namespace, values, option_string=None):
            existing_values = getattr(namespace, self.dest, None)
            all_values = _ConcatList(existing_values, values)
            if dedup:
                deduped_values = []
                for value in all_values:
                    if value not in deduped_values:
                        deduped_values.append(value)
                all_values = deduped_values
            setattr(namespace, self.dest, all_values)
    return Action