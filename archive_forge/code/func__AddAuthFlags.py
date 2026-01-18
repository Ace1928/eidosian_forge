from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib import tasks as tasks_api_lib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddAuthFlags(parser):
    """Add flags for http auth."""
    auth_group = parser.add_mutually_exclusive_group(help='            How the request sent to the target when executing the task should be\n            authenticated.\n            ')
    oidc_group = auth_group.add_argument_group(help='OpenId Connect')
    oidc_group.add_argument('--oidc-service-account-email', required=True, help="            The service account email to be used for generating an OpenID\n            Connect token to be included in the request sent to the target when\n            executing the task. The service account must be within the same\n            project as the queue. The caller must have\n            'iam.serviceAccounts.actAs' permission for the service account.\n            ")
    oidc_group.add_argument('--oidc-token-audience', help='            The audience to be used when generating an OpenID Connect token to\n            be included in the request sent to the target when executing the\n            task. If not specified, the URI specified in the target will be\n            used.\n            ')
    oauth_group = auth_group.add_argument_group(help='OAuth2')
    oauth_group.add_argument('--oauth-service-account-email', required=True, help="            The service account email to be used for generating an OAuth2 access\n            token to be included in the request sent to the target when\n            executing the task. The service account must be within the same\n            project as the queue. The caller must have\n            'iam.serviceAccounts.actAs' permission for the service account.\n            ")
    oauth_group.add_argument('--oauth-token-scope', help="            The scope to be used when generating an OAuth2 access token to be\n            included in the request sent to the target when executing the task.\n            If not specified, 'https://www.googleapis.com/auth/cloud-platform'\n            will be used.\n            ")