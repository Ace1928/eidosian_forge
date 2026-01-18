from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import network_troubleshooter
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute import user_permission_troubleshooter
from googlecloudsdk.command_lib.compute import vm_boot_troubleshooter
from googlecloudsdk.command_lib.compute import vm_status_troubleshooter
from googlecloudsdk.command_lib.compute import vpc_troubleshooter
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.util.ssh import containers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
def AddContainerArg(parser):
    parser.add_argument('--container', help='          The name or ID of a container inside of the virtual machine instance\n          to connect to. This only applies to virtual machines that are using\n          a Google Container-Optimized virtual machine image. For more\n          information, see [](https://cloud.google.com/compute/docs/containers).\n          ')