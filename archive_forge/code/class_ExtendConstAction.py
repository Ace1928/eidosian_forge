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
class ExtendConstAction(argparse.Action):
    """Extends the dest arg with a constant list."""

    def __init__(self, *args, **kwargs):
        super(ExtendConstAction, self).__init__(*args, nargs=0, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        """Extends the dest with the const list."""
        cur = getattr(self, self.dest, [])
        setattr(namespace, self.dest, cur + self.const)