from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.auth import exceptions as auth_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.command_lib.auth import auth_util
from googlecloudsdk.command_lib.auth import flags
from googlecloudsdk.command_lib.config import config_helper
from googlecloudsdk.core import config
from googlecloudsdk.core.credentials import store as c_store
from oauth2client import client
class IdentityToken(base.Command):
    """Print an identity token for the specified account."""
    detailed_help = {'DESCRIPTION': '        {description}\n        ', 'EXAMPLES': '        To print identity tokens:\n\n          $ {command}\n\n        To print identity token for account \'foo@example.com\' whose audience\n        is \'https://service-hash-uc.a.run.app\', run:\n\n          $ {command} foo@example.com --audiences="https://service-hash-uc.a.run.app"\n\n        To print identity token for an impersonated service account \'my-account@example.iam.gserviceaccount.com\'\n        whose audience is \'https://service-hash-uc.a.run.app\', run:\n\n          $ {command} --impersonate-service-account="my-account@example.iam.gserviceaccount.com" --audiences="https://service-hash-uc.a.run.app"\n\n        To print identity token of a Compute Engine instance, which includes\n        project and instance details as well as license codes for images\n        associated with the instance, run:\n\n          $ {command} --token-format=full --include-license\n\n        To print identity token for an impersonated service account\n        \'my-account@example.iam.gserviceaccount.com\', which includes the email\n        address of the service account, run:\n\n          $ {command} --impersonate-service-account="my-account@example.iam.gserviceaccount.com" --include-email\n        '}

    @staticmethod
    def Args(parser):
        flags.AddAccountArg(parser)
        flags.AddAudienceArg(parser)
        flags.AddGCESpecificArgs(parser)
        flags.AddIncludeEmailArg(parser)
        parser.display_info.AddFormat('value(id_token)')

    @c_exc.RaiseErrorInsteadOf(auth_exceptions.AuthenticationError, client.Error)
    def Run(self, args):
        """Run the print_identity_token command."""
        credential = _Run(args)
        return credential