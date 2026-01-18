from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import ipaddr
import six
def ParseErrors(errors):
    """Parses errors to prepare the right error contents."""
    filtered_errors = []
    for error in errors:
        if not hasattr(error, 'message'):
            filtered_errors.append(error)
        elif IsQuotaExceededError(error):
            filtered_errors.append(CreateQuotaExceededMsg(error))
        elif ShouldUseYaml(error):
            filtered_errors.append(error)
        else:
            filtered_errors.append(error.message)
    return filtered_errors