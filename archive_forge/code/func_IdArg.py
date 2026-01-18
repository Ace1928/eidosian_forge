from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def IdArg(description):
    return base.Argument('id', metavar='ORGANIZATION_ID', completer=completers.OrganizationCompleter, help='ID or domain for the organization {0}'.format(description))