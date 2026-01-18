from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import ipaddress
import re
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import preset_profiles
from googlecloudsdk.command_lib.privateca import text_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from that bucket.
def ParseIdentityConstraints(args):
    """Parses the identity flags into a CertificateIdentityConstraints object."""
    messages = privateca_base.GetMessagesModule('v1')
    return messages.CertificateIdentityConstraints(allowSubjectPassthrough=args.copy_subject, allowSubjectAltNamesPassthrough=args.copy_sans, celExpression=messages.Expr(expression=args.identity_cel_expression) if args.IsSpecified('identity_cel_expression') else None)