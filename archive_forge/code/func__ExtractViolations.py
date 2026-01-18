from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import logging
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
def _ExtractViolations(self, details):
    """Extracts a map of violations from the given error's details.

    Args:
      details: JSON-parsed details field from parsed json of error.

    Returns:
      Map[str, str] sub -> error description. The iterator of it is ordered by
      the order the subjects first appear in the errror.
    """
    results = collections.OrderedDict()
    for detail in details:
        if 'violations' not in detail:
            continue
        violations = detail['violations']
        if not isinstance(violations, list):
            continue
        sub = detail.get('subject')
        for violation in violations:
            try:
                local_sub = violation.get('subject')
                subject = sub or local_sub
                if subject:
                    if subject in results:
                        results[subject] += '\n' + violation['description']
                    else:
                        results[subject] = violation['description']
            except (KeyError, TypeError):
                pass
    return results