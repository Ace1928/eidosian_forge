from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
def AddConstraintArgument(parser):
    parser.add_argument('--constraint', metavar='CONSTRAINT', required=True, help='        The name of the constraint to analyze organization policies for. The\n        response only contains analyzed organization policies for the provided\n        constraint.\n\n        Example:\n\n        * organizations/{ORGANIZATION_NUMBER}/customConstraints/{CUSTOM_CONSTRAINT_NAME}\n          for a user-defined custom constraint.\n        ')