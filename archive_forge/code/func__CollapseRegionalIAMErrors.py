from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.command_lib.bms import util
import six
def _CollapseRegionalIAMErrors(errors):
    """If all errors are PERMISSION_DENIEDs, use a single global error instead."""
    if errors:
        matches = [_REGIONAL_IAM_REGEX.match(e) for e in errors]
        if all((match is not None for match in matches)) and len(set((match.group(1) for match in matches))) == 1:
            errors = ['PERMISSION_DENIED: Permission %s denied on projects/%s' % (matches[0].group(1), matches[0].group(2))]
    return errors