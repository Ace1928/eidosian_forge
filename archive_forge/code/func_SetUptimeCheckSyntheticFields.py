from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def SetUptimeCheckSyntheticFields(args, messages, uptime_check):
    """Set Synthetic Monitor fields based on args."""
    uptime_check.syntheticMonitor = messages.SyntheticMonitorTarget()
    uptime_check.syntheticMonitor.cloudFunctionV2 = messages.CloudFunctionV2Target()
    uptime_check.syntheticMonitor.cloudFunctionV2.name = args.synthetic_target