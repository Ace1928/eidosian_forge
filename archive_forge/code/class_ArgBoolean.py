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
class ArgBoolean(ArgType):
    """Interpret an argument value as a bool."""

    def __init__(self, truthy_strings=None, falsey_strings=None, case_sensitive=False):
        self._case_sensitive = case_sensitive
        if truthy_strings:
            self._truthy_strings = truthy_strings
        else:
            self._truthy_strings = ['true', 'yes']
        if falsey_strings:
            self._falsey_strings = falsey_strings
        else:
            self._falsey_strings = ['false', 'no']

    def __call__(self, arg_value):
        if not self._case_sensitive:
            normalized_arg_value = arg_value.lower()
        else:
            normalized_arg_value = arg_value
        if normalized_arg_value in self._truthy_strings:
            return True
        if normalized_arg_value in self._falsey_strings:
            return False
        raise ArgumentTypeError('Invalid flag value [{0}], expected one of [{1}]'.format(arg_value, ', '.join(self._truthy_strings + self._falsey_strings)))