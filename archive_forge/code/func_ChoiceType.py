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
def ChoiceType(raw_value):
    if element_type:
        typed_value = element_type(raw_value)
    else:
        typed_value = raw_value
    if typed_value not in choices:
        raise ArgumentTypeError('{value} must be one of [{choices}]'.format(value=typed_value, choices=', '.join([six.text_type(choice) for choice in self.visible_choices])))
    return typed_value