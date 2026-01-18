from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def ParseColumnFamilies(family_list):
    """Parses column families value object from the string list.

  Args:
    family_list: A list that contains one or more strings representing family
      name and optional GC rules in the format of `family_name[:gc_rule]`, such
      as `my_family_1,my_family_2:maxage=10d`.

  Returns:
    A column families value object.
  """
    results = []
    for expr in family_list:
        family, gc_rule = ParseExpr(expr)
        column_family = util.GetAdminMessages().ColumnFamily(gcRule=gc_rule)
        results.append(util.GetAdminMessages().Table.ColumnFamiliesValue.AdditionalProperty(key=family, value=column_family))
    return util.GetAdminMessages().Table.ColumnFamiliesValue(additionalProperties=results)