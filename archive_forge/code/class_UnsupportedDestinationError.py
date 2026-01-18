from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.eventarc import triggers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import flags
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class UnsupportedDestinationError(exceptions.Error):
    """Error when none of the supported destination args is specified."""