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
def _DeleteTypeAbbr(suffix, type_abbr='B'):
    """Returns suffix with trailing type abbreviation deleted."""
    if not suffix:
        return suffix
    s = suffix.upper()
    i = len(s)
    for c in reversed(type_abbr.upper()):
        if not i:
            break
        if s[i - 1] == c:
            i -= 1
    return suffix[:i]