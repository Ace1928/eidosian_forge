import logging
import re
from oslo_policy import _checks
def _parse_list_rule(rule):
    """Translates the old list-of-lists syntax into a tree of Check objects.

    Provided for backwards compatibility.
    """
    if not rule:
        return _checks.TrueCheck()
    or_list = []
    for inner_rule in rule:
        if not inner_rule:
            continue
        if isinstance(inner_rule, str):
            inner_rule = [inner_rule]
        and_list = [_parse_check(r) for r in inner_rule]
        if len(and_list) == 1:
            or_list.append(and_list[0])
        else:
            or_list.append(_checks.AndCheck(and_list))
    if not or_list:
        return _checks.FalseCheck()
    elif len(or_list) == 1:
        return or_list[0]
    return _checks.OrCheck(or_list)