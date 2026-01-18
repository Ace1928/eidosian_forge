from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import os
import re
import sys
import types
import uuid
import argcomplete
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import backend
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
def _HandleAllErrors(self, exc, command_path_string, specified_arg_names):
    """Handle all errors.

    Args:
      exc: Exception, The exception that was raised.
      command_path_string: str, The '.' separated command path.
      specified_arg_names: [str], The specified arg named scrubbed for metrics.

    Raises:
      exc or a core.exceptions variant that does not produce a stack trace.
    """
    error_extra_info = {'error_code': getattr(exc, 'exit_code', 1)}
    http_status_code = getattr(getattr(exc, 'payload', None), 'status_code', None)
    if http_status_code is not None:
        error_extra_info['http_status_code'] = http_status_code
    metrics.Commands(command_path_string, config.CLOUD_SDK_VERSION, specified_arg_names, error=exc.__class__, error_extra_info=error_extra_info)
    metrics.Error(command_path_string, exc.__class__, specified_arg_names, error_extra_info=error_extra_info)
    exceptions.HandleError(exc, command_path_string, self.__known_error_handler)