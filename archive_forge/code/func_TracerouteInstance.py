from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import os
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.diagnose import external_helper
from googlecloudsdk.command_lib.compute.diagnose import internal_helpers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def TracerouteInstance(self, instance, traceroute_args, dry_run, resource_registry):
    """Runs a traceroute from localhost to a GCE VM.

    Args:
      instance: Compute Engine VM.
      traceroute_args: Additional traceroute args to be passed on.
      dry_run: Whether to only print commands instead of running them.
      resource_registry: gcloud class used for obtaining data from the
        resources.
    """
    instance_string = internal_helpers.GetInstanceNetworkTitleString(instance)
    log.out.Print('>>> Tracerouting to %s' % instance_string)
    external_ip = ssh_utils.GetExternalIPAddress(instance)
    cmd = ['traceroute', external_ip]
    if traceroute_args:
        cmd += traceroute_args
    if dry_run:
        external_helper.DryRunLog(' '.join(cmd))
    else:
        external_helper.RunSubprocess(proc_name='Traceroute', command_list=cmd)
        log.out.Print('>>>')