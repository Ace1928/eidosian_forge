from apitools.base.py import list_pager
from googlecloudsdk.api_lib.quotas import message_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import common_args
def _GetIgnoreSafetyChecks(args, request):
    ignore_safety_checks = []
    if args.allow_quota_decrease_below_usage:
        ignore_safety_checks.append(request.IgnoreSafetyChecksValueValuesEnum.QUOTA_DECREASE_BELOW_USAGE)
    if args.allow_high_percentage_quota_decrease:
        ignore_safety_checks.append(request.IgnoreSafetyChecksValueValuesEnum.QUOTA_DECREASE_PERCENTAGE_TOO_HIGH)
    return ignore_safety_checks