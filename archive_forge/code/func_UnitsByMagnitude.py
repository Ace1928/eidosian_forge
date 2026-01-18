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
def UnitsByMagnitude(suggested_binary_size_scales=None):
    """Returns a list of the units in scales sorted by magnitude."""
    scale_items = sorted(six.iteritems(scales), key=lambda value: (value[1], value[0]))
    if suggested_binary_size_scales is None:
        return [key + type_abbr for key, _ in scale_items]
    return [key + type_abbr for key, _ in scale_items if key + type_abbr in suggested_binary_size_scales]