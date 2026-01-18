from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def ParseExpr(expr):
    """Parses family name and GC rules from the string expression.

  Args:
    expr: A string express contains family name and optional GC rules in the
    format of `family_name[:gc_rule]`, such as `my_family:maxage=10d`.

  Returns:
    A family name and a GcRule object defined in the Bigtable admin API.

  Raises:
    BadArgumentExpection: the input string is mal-formatted.
  """
    expr_list = expr.split(':')
    family = expr_list[0]
    expr_list_len = len(expr_list)
    if expr_list_len > 2 or family != family.strip():
        raise exceptions.BadArgumentException('--column-families', 'Input column family ({0}) is mal-formatted.'.format(expr))
    if expr_list_len == 1:
        return (family, None)
    if not expr_list[1]:
        raise exceptions.BadArgumentException('--column-families', 'Input column family ({0}) is mal-formatted.'.format(expr))
    gc_rule = expr_list[1]
    union_list = gc_rule.split('||')
    intersection_list = gc_rule.split('&&')
    if len(union_list) == 2 and len(intersection_list) == 1:
        return (family, util.GetAdminMessages().GcRule(union=util.GetAdminMessages().Union(rules=ParseBinaryRule(union_list))))
    elif len(union_list) == 1 and len(intersection_list) == 2:
        return (family, util.GetAdminMessages().GcRule(intersection=util.GetAdminMessages().Intersection(rules=ParseBinaryRule(intersection_list))))
    elif len(union_list) == 1 and len(intersection_list) == 1:
        if gc_rule:
            return (family, ParseSingleRule(gc_rule))
    else:
        raise exceptions.BadArgumentException('--column-families', 'Input column family ({0}) is mal-formatted.'.format(expr))