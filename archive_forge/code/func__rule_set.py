from __future__ import absolute_import, division, print_function
from . import utils
def _rule_set(rules):
    return set(((frozenset(r.get('verbs', []) or []), frozenset(r.get('resources', []) or []), frozenset(r.get('resource_names', []) or [])) for r in rules))