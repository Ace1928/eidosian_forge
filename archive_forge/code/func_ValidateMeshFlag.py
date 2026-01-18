from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
def ValidateMeshFlag(args):
    """Validates the values of the --mesh flag."""
    if getattr(args, 'mesh', False):
        if args.no_scopes:
            raise exceptions.ConflictingArgumentsException('--mesh', '--no-scopes')
        rgx = '(.*)\\/(.*)'
        try:
            if not re.match(rgx, args.mesh['gke-cluster']):
                raise ValueError
        except ValueError:
            raise exceptions.InvalidArgumentException('gke-cluster', 'GKE cluster value should have the format location/name.')
        try:
            if not re.match(rgx, args.mesh['workload']):
                raise ValueError
        except ValueError:
            raise exceptions.InvalidArgumentException('workload', 'Workload value should have the format namespace/name.')