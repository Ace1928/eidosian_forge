from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import flags as scc_flags
from googlecloudsdk.command_lib.scc import util as scc_util
from googlecloudsdk.command_lib.scc.findings import flags
from googlecloudsdk.command_lib.scc.findings import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import times
def _UpdateEventTime(args, req, version):
    """Process and include the event time of a finding."""
    if args.event_time:
        event_time_dt = times.ParseDateTime(args.event_time)
        if version == 'v1':
            req.finding.eventTime = times.FormatDateTime(event_time_dt)
        else:
            req.googleCloudSecuritycenterV2Finding.eventTime = times.FormatDateTime(event_time_dt)
    if args.event_time is None:
        event_time = datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        if version == 'v1':
            if req.finding is None:
                req.finding = securitycenter_client.GetMessages().Finding()
            req.finding.eventTime = event_time
        else:
            if req.googleCloudSecuritycenterV2Finding is None:
                req.googleCloudSecuritycenterV2Finding = securitycenter_client.GetMessages().GoogleCloudSecuritycenterV2Finding()
            req.googleCloudSecuritycenterV2Finding.eventTime = event_time
        req.updateMask = req.updateMask + ',event_time'
    return req