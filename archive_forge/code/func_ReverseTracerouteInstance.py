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
def ReverseTracerouteInstance(self, instance, user, external_route_ip, traceroute_args, dry_run, resource_registry):
    """Runs a traceroute from a GCE VM to localhost.

    Args:
      instance: Compute Engine VM.
      user: The user to use to SSH into the instance.
      external_route_ip: the ip to which traceroute from the VM
      traceroute_args: Additional traceroute args to be passed on.
      dry_run: Whether to only print commands instead of running them.
      resource_registry: gcloud class used for obtaining data from the
        resources.
    Raises:
      ssh.CommandError: there was an error running a SSH command
    """
    instance_string = internal_helpers.GetInstanceNetworkTitleString(instance)
    log.out.Print('<<< Reverse tracerouting from %s' % instance_string)
    log.out.flush()
    if dry_run:
        external_route_ip = '<SELF-IP>'
    cmd = ['traceroute', external_route_ip]
    if traceroute_args:
        cmd += traceroute_args
    external_helper.RunSSHCommandToInstance(command_list=cmd, instance=instance, user=user, args=self._args, ssh_helper=self._ssh_helper, dry_run=dry_run)
    if not dry_run:
        log.out.Print('<<<')