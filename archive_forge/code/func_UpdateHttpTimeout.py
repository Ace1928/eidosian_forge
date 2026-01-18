from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core.util import times
def UpdateHttpTimeout(args, function, api_version, release_track):
    """Update core/http_timeout using args and function timeout.

  Args:
    args: The arguments from the command line parser
    function: function definition
    api_version: v1 or v2
    release_track: ALPHA, BETA, or GA
  """
    if release_track in [base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA]:
        timeout = 540 if api_version == 'v1' else 3600
        if args.timeout:
            timeout = int(args.timeout)
        elif api_version == 'v1' and function.timeout:
            timeout = int(times.ParseDuration(function.timeout, default_suffix='s').total_seconds + 30)
        elif api_version == 'v2' and function.serviceConfig and function.serviceConfig.timeoutSeconds:
            timeout = int(function.serviceConfig.timeoutSeconds) + 30
        properties.VALUES.core.http_timeout.Set(timeout)