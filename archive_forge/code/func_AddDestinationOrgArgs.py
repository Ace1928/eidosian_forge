from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
def AddDestinationOrgArgs(parser):
    parser.add_argument('--destination-organization', metavar='ORGANIZATION_ID', required=False, help='The destination organization ID to perform the analysis.')