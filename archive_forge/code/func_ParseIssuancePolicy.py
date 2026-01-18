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
def ParseIssuancePolicy(args):
    """Parses an IssuancePolicy proto message from the args."""
    if not args.IsSpecified('issuance_policy'):
        return None
    try:
        return messages_util.DictToMessageWithErrorCheck(args.issuance_policy, privateca_base.GetMessagesModule('v1').IssuancePolicy)
    except (messages_util.DecodeError, AttributeError):
        raise exceptions.InvalidArgumentException('--issuance-policy', 'Unrecognized field in the Issuance Policy.')