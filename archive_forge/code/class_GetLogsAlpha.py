from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.functions.v1 import util as util_v1
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.logging import common as logging_common
from googlecloudsdk.api_lib.logging import util as logging_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class GetLogsAlpha(GetLogsBeta):
    """Display log entries produced by Google Cloud Functions."""