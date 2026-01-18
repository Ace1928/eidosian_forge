from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
import six
Builds an ExecutionConfig instance.

    Build a ExecutionConfig instance according to user settings.
    Returns None if all fileds are None.

    Args:
      args: Parsed arguments.

    Returns:
      ExecutionConfig: A ExecutionConfig instance. None if all fields are
      None.
    