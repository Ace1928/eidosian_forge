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
class _RoutesArgs(object):
    """Helper to setting and getting values for the args."""

    @classmethod
    def Args(cls, parser):
        """Creates the flags stmts for the command."""
        ssh_utils.BaseSSHCLIHelper.Args(parser)
        base_classes.ZonalLister.Args(parser)
        parser.add_argument('--container', help='            The name or ID of a container inside of the virtual machine instance\n            to connect to. This only applies to virtual machines that are using\n            a Container-Optimized OS virtual machine image.\n            For more information, see\n            [](https://cloud.google.com/compute/docs/containers)\n            ')
        parser.add_argument('--external-route-ip', default=None, help='For reverse traceroute, this will be the ip given to the VM instance to traceroute to. This will override all obtained ips.')
        parser.add_argument('--reverse-traceroute', action='store_true', help='If enabled, will also run traceroute from the VM to the host')
        parser.add_argument('--ssh-flag', action='append', help='        Additional flags to be passed to *ssh(1)*. It is recommended that flags\n        be passed using an assignment operator and quotes. This flag will\n        replace occurences of ``%USER%\'\' and ``%INSTANCE%\'\' with their\n        dereferenced values. Example:\n\n          $ {command} example-instance --zone us-central1-a           --ssh-flag="-vvv" --ssh-flag="-L 80:%INSTANCE%:80"\n\n        is equivalent to passing the flags ``--vvv\'\' and ``-L\n        80:162.222.181.197:80\'\' to *ssh(1)* if the external IP address of\n        \'example-instance\' is 162.222.181.197.\n        ')
        parser.add_argument('--user', help='        User for login to the selected VMs.\n        If not specified, the default user will be used.\n        ')
        parser.add_argument('traceroute_args', nargs=argparse.REMAINDER, help='            Flags and positionals passed to the underlying traceroute call.\n            ', example='            $ {command} example-instance -- -w 0.5 -q 5 42\n        ')

    @classmethod
    def GetFilters(cls, args):
        filters = []
        if args.regexp:
            filters.append('name eq %s' % args.regexp)
        if not filters:
            return None
        filters = ' AND '.join(filters)
        return filters