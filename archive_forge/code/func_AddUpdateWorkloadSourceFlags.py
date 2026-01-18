from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import Collection
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
def AddUpdateWorkloadSourceFlags(parser):
    """Adds the flags for update workload source command."""
    parser.add_argument('--add-resources', type=arg_parsers.ArgList(), help='A list of allowed resources to add to the workload source.', metavar='RESOURCE')
    parser.add_argument('--add-attached-service-accounts', type=arg_parsers.ArgList(), help='A list of allowed attached_service_accounts to add to the workload source.', metavar='SERVICE_ACCOUNT')
    parser.add_argument('--remove-resources', type=arg_parsers.ArgList(), help='A list of allowed resources to remove from the workload source.', metavar='RESOURCE')
    parser.add_argument('--remove-attached-service-accounts', type=arg_parsers.ArgList(), help='A list of allowed attached_service_accounts to remove from the workload source.', metavar='SERVICE_ACCOUNT')
    parser.add_argument('--clear-resources', help='Remove all the allowed resources for the workload source.', action='store_true')
    parser.add_argument('--clear-attached-service-accounts', help='Remove all the allowed attached_service_accounts for the workload source.', action='store_true')