from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateClearBackends(unused_ref, args, update_federation_req):
    """Validate if users run update federation command with --clear-backends arg only.

  Args:
    unused_ref: A resource ref to the parsed Federation resource.
    args: The parsed args namespace from CLI.
    update_federation_req: The request for the API call.

  Returns:
    String request
  Raises:
    BadArgumentException: When users run update federation command with
    --clear-backends arg only.
  """
    args_set = set(args.GetSpecifiedArgNames())
    if '--clear-backends' in args_set and '--update-backends' not in args_set:
        raise exceptions.BadArgumentException('--clear-backends', '--clear-backends must be used with --update-backends')
    return update_federation_req