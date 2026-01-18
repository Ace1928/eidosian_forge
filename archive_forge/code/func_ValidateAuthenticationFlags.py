from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kuberun.core import events_constants
from googlecloudsdk.calliope import exceptions as calliope_exceptions
def ValidateAuthenticationFlags(args):
    """Validate authentication mode secrets includes --copy-default-secret."""
    if args.authentication and args.authentication == 'secrets':
        if not args.copy_default_secret:
            raise calliope_exceptions.RequiredArgumentException('--copy-default-secret', 'Secrets authentication mode missing flag.')
    elif args.copy_default_secret:
        raise calliope_exceptions.InvalidArgumentException('--copy-default-secret', 'Only secrets authentication mode uses desired flag.')