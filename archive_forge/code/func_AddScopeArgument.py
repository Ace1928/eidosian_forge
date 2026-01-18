from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
def AddScopeArgument(parser):
    parser.add_argument('--scope', metavar='SCOPE', required=True, help="        Scope can only be an organization. The analysis is\n        limited to the Cloud organization policies within this scope. The caller must be\n        granted the `cloudasset.assets.searchAllResources` permission on\n        the desired scope.\n\n        The allowed values are:\n\n          * ```organizations/{ORGANIZATION_NUMBER}``` (e.g. ``organizations/123456'')\n        ")