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
def ValidateIdentityConstraints(args, existing_copy_subj=False, existing_copy_sans=False, for_update=False):
    """Validates the template identity constraints flags.

  Args:
    args: the parser for the flag. Expected to have copy_sans and copy_subject
      registered as flags
    existing_copy_subj: A pre-existing value for the subject value, if
      applicable.
    existing_copy_sans: A pre-existing value for the san value, if applicable.
    for_update: Whether the validation is for an update to a template.
  """
    copy_san = args.copy_sans or (not args.IsSpecified('copy_sans') and existing_copy_sans)
    copy_subj = args.copy_subject or (not args.IsSpecified('copy_subject') and existing_copy_subj)
    if for_update:
        missing_identity_conf_msg = 'The resulting updated template will have no subject or SAN passthroughs. '
    else:
        missing_identity_conf_msg = 'Neither copy-sans nor copy-subject was specified. '
    missing_identity_conf_msg += 'This means that all certificate requests that use this template must use identity reflection.'
    if not copy_san and (not copy_subj) and (not console_io.PromptContinue(message=missing_identity_conf_msg, default=True)):
        raise privateca_exceptions.UserAbortException('Aborted by user.')