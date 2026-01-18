from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.policy_intelligence import policy_analyzer
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def _ArgsAlpha(parser):
    """Parses arguments for the commands."""
    parser.add_argument('--activity-type', required=True, type=str, choices=['serviceAccountLastAuthentication', 'serviceAccountKeyLastAuthentication', 'dailyAuthorization'], help='Type of the activities.\n      ')
    parser.add_mutually_exclusive_group(required=True).add_argument('--project', type=str, help='The project ID or number to query the activities.\n      ')
    parser.add_argument('--query-filter', type=str, default='', help='Filter on activities. \n\n      For last authentication activities, this field is separated by "OR" if multiple filters are specified. At most 10 filter restrictions are supported in the query-filter. \n\n        e.g. --query-filter=\'activities.full_resource_name="//iam.googleapis.com/projects/project-id/serviceAccounts/service-account-name-1@project-id.iam.gserviceaccount.com" OR activities.full_resource_name="//iam.googleapis.com/projects/project-id/serviceAccounts/service-account-name-2@project-id.iam.gserviceaccount.com"\'\n\n      For daily authorization activities, this field is separated by "OR" and "AND". At most 10 filter restrictions per layer and at most 2 layers are supported in the query-filter. \n\n        e.g. --query-filter=\'activities.activity.date="2022-01-01" AND activities.activity.permission="spanner.databases.list" AND (activities.activity.principal="principal_1@your-organization.com" OR activities.activity.principal="principal_2@your-organization.com")\'')
    parser.add_argument('--limit', type=arg_parsers.BoundedInt(1, sys.maxsize, unlimited=True), default=1000, help='Max number of query result. Default to be 1000 and max to be unlimited, i.e., --limit=unlimited.')
    parser.add_argument('--page-size', type=arg_parsers.BoundedInt(1, 1000), default=500, help='Max page size for each http response. Default to be 500 and max to be 1000.')