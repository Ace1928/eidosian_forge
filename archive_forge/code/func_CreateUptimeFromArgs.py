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
def CreateUptimeFromArgs(args, messages):
    """Builds an Uptime message from args."""
    uptime_base_flags = ['--resource-labels', '--group-id', '--synthetic-target']
    ValidateAtleastOneSpecified(args, uptime_base_flags)
    uptime_check = messages.UptimeCheckConfig()
    if args.IsSpecified('resource_labels'):
        SetUptimeCheckMonitoredResourceFields(args, messages, uptime_check)
    elif args.IsSpecified('group_id'):
        SetUptimeCheckGroupFields(args, messages, uptime_check)
    else:
        SetUptimeCheckSyntheticFields(args, messages, uptime_check)
    user_labels = None
    if args.IsSpecified('user_labels'):
        user_labels = messages.UptimeCheckConfig.UserLabelsValue()
        for k, v in args.user_labels.items():
            user_labels.additionalProperties.append(messages.UptimeCheckConfig.UserLabelsValue.AdditionalProperty(key=k, value=v))
    headers = None
    if args.IsSpecified('headers'):
        headers = messages.HttpCheck.HeadersValue()
        if headers is not None:
            for k, v in args.headers.items():
                headers.additionalProperties.append(messages.HttpCheck.HeadersValue.AdditionalProperty(key=k, value=v))
    uptime_check.timeout = '60s'
    uptime_check.period = '60s'
    ModifyUptimeCheck(uptime_check, messages, args, regions=args.regions, user_labels=user_labels, headers=headers, status_classes=args.status_classes, status_codes=args.status_codes)
    return uptime_check