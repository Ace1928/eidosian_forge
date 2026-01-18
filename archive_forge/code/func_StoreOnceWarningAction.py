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
def StoreOnceWarningAction(flag_name):
    """Emits a warning message when a flag is specified more than once.

  The created action is similar to StoreOnceAction. The difference is that
  this action prints a warning message instead of raising an exception when the
  flag is specified more than once. Because it is a breaking change to switch an
  existing flag to StoreOnceAction, StoreOnceWarningAction can be used in the
  deprecation period.

  Args:
    flag_name: The name of the flag to apply this action on.

  Returns:
    An Action class.
  """

    class Action(argparse.Action):
        """Emits a warning message when a flag is specified more than once."""

        def OnSecondArgumentPrintWarning(self):
            log.warning('"{0}" argument is specified multiple times which will be disallowed in future versions. Please only specify it once.'.format(flag_name))

        def __init__(self, *args, **kwargs):
            self.dest_is_populated = False
            super(Action, self).__init__(*args, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            if self.dest_is_populated:
                self.OnSecondArgumentPrintWarning()
            self.dest_is_populated = True
            setattr(namespace, self.dest, values)
    return Action