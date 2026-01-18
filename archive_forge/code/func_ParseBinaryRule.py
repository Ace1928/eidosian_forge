from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def ParseBinaryRule(rule_list):
    """Parses GC rules from a rule list of 2 elements.

  Args:
    rule_list: A string list containing 2 elements.

  Returns:
    A list of GcRule objects.

  Raises:
    BadArgumentExpection: the input list is mal-formatted.
  """
    if len(rule_list) != 2:
        raise exceptions.BadArgumentException('--column-families', 'Invalid union or intersection rule: ' + ' '.join(rule_list))
    results = []
    for rule in rule_list:
        results.append(ParseSingleRule(rule))
    return results