from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def _AddMaxAttemptsFieldsFromArgs(args, config_object, is_alpha=False):
    if args.IsSpecified('max_attempts'):
        if args.max_attempts is None:
            if is_alpha:
                config_object.unlimitedAttempts = True
            else:
                config_object.maxAttempts = -1
        else:
            config_object.maxAttempts = args.max_attempts