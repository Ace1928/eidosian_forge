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
def ParseSubjectKeyId(args, messages):
    """Parses the subject key id for input into CertificateConfig.

  Args:
    args: The parsed argument values
    messages: PrivateCA's messages modules

  Returns:
    A CertificateConfigKeyId message object
  """
    if not args.IsSpecified('subject_key_id'):
        return None
    skid = args.subject_key_id
    if not re.match(_SKID_REGEX, skid):
        raise exceptions.InvalidArgumentException('--subject-key-id', 'Subject key id must be an even length lowercase hex string.')
    return messages.CertificateConfigKeyId(keyId=skid)