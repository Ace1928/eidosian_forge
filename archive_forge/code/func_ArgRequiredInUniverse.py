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
def ArgRequiredInUniverse(default_universe: bool=False, non_default_universe: bool=True) -> bool:
    """Determines if the arg is required based on the universe domain.

  Args:
    default_universe: Whether the arg is required in the default universe.
      Defaults to False.
    non_default_universe: Whether the arg is required outside of the default
      universe. Defaults to True.

  Returns:
    bool, whether the arg is required in the current universe.
  """
    if properties.IsDefaultUniverse():
        return default_universe
    return non_default_universe