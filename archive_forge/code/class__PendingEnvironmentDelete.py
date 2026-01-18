from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
class _PendingEnvironmentDelete(object):
    """Data class holding information about a pending environment deletion."""

    def __init__(self, environment_name, operation):
        self.environment_name = environment_name
        self.operation = operation