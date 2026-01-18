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
def _AddHttpTargetAuthFlags(parser=None, is_email_required=False):
    """Add flags for http auth."""
    auth_group = base.ArgumentGroup(mutex=True, help='            If specified, all `Authorization` headers in the HttpRequest.headers\n            field will be overridden for any tasks executed on this queue.\n            ')
    oidc_group = base.ArgumentGroup(help='OpenId Connect')
    oidc_email_arg = base.Argument('--http-oidc-service-account-email-override', required=is_email_required, help="            The service account email to be used for generating an OpenID\n            Connect token to be included in the request sent to the target when\n            executing the task. The service account must be within the same\n            project as the queue. The caller must have\n            'iam.serviceAccounts.actAs' permission for the service account.\n            ")
    oidc_group.AddArgument(oidc_email_arg)
    oidc_token_arg = base.Argument('--http-oidc-token-audience-override', help='            The audience to be used when generating an OpenID Connect token to\n            be included in the request sent to the target when executing the\n            task. If not specified, the URI specified in the target will be\n            used.\n            ')
    oidc_group.AddArgument(oidc_token_arg)
    oauth_group = base.ArgumentGroup(help='OAuth2')
    oauth_email_arg = base.Argument('--http-oauth-service-account-email-override', required=is_email_required, help="            The service account email to be used for generating an OAuth2 access\n            token to be included in the request sent to the target when\n            executing the task. The service account must be within the same\n            project as the queue. The caller must have\n            'iam.serviceAccounts.actAs' permission for the service account.\n            ")
    oauth_group.AddArgument(oauth_email_arg)
    oauth_scope_arg = base.Argument('--http-oauth-token-scope-override', help="            The scope to be used when generating an OAuth2 access token to be\n            included in the request sent to the target when executing the task.\n            If not specified, 'https://www.googleapis.com/auth/cloud-platform'\n            will be used.\n            ")
    oauth_group.AddArgument(oauth_scope_arg)
    auth_group.AddArgument(oidc_group)
    auth_group.AddArgument(oauth_group)
    if parser is not None:
        auth_group.AddToParser(parser)
    return [oidc_email_arg, oidc_token_arg, oauth_email_arg, oauth_scope_arg]