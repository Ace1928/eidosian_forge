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
def SetUptimeCheckGroupFields(args, messages, uptime_check):
    """Set Group fields based on args."""
    group_mapping = {'gce-instance': 'INSTANCE', 'aws-elb-load-balancer': 'AWS_ELB_LOAD_BALANCER', None: 'INSTANCE'}
    uptime_check.resourceGroup = messages.ResourceGroup()
    uptime_check.resourceGroup.groupId = args.group_id
    uptime_check.resourceGroup.resourceType = arg_utils.ChoiceToEnum(group_mapping.get(args.group_type), messages.ResourceGroup.ResourceTypeValueValuesEnum, item_type='group type')