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
def StoreFilePathAndContentsAction(binary=False):
    """Returns Action that stores both file content and file path.

  Args:
   binary: boolean, whether or not this is a binary file.

  Returns:
   An argparse action.
  """

    class Action(argparse.Action):
        """Stores both file content and file path.

      Stores file contents under original flag DEST and stores file path under
      DEST_path.
    """

        def __init__(self, *args, **kwargs):
            super(Action, self).__init__(*args, **kwargs)

        def __call__(self, parser, namespace, value, option_string=None):
            """Stores the contents of the file and the file name in namespace."""
            try:
                content = console_io.ReadFromFileOrStdin(value, binary=binary)
            except files.Error as e:
                raise ArgumentTypeError(e)
            setattr(namespace, self.dest, content)
            new_dest = '{}_path'.format(self.dest)
            setattr(namespace, new_dest, value)
    return Action