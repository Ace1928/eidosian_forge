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
class StoreOnceAction(argparse.Action):
    """Action that disallows repeating a flag.

  When using action='store' (the default), argparse allows multiple instances of
  a flag to be specified with the last one determining the value and the rest
  silently dropped. This is often undesirable if the command accepts only one
  value but users try to repeat the flag (either accidentally, or when
  mistakenly expecting the repeated values to be appended or merged somehow).

  In such cases, one can instead use StoreOnceAction which disallows specifying
  the same flag multiple times. So for instance, providing:

    --foo 123 --foo 456

  will result in an error stating that --foo cannot be specified more than once.
  """

    def OnSecondArgumentRaiseError(self):
        raise argparse.ArgumentError(self, _GenerateErrorMessage('"{0}" argument cannot be specified multiple times'.format(self.dest)))

    def __init__(self, *args, **kwargs):
        self.dest_is_populated = False
        super(StoreOnceAction, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if self.dest_is_populated:
            self.OnSecondArgumentRaiseError()
        self.dest_is_populated = True
        setattr(namespace, self.dest, values)