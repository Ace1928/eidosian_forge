import sys
import json
from .symbols import *
from .symbols import Symbol
class ExplicitJsonDiffSyntax(object):

    def emit_set_diff(self, a, b, s, added, removed):
        if s == 0.0 or len(removed) == len(a):
            return b
        else:
            d = {}
            if removed:
                d[discard] = removed
            if added:
                d[add] = added
            return d

    def emit_list_diff(self, a, b, s, inserted, changed, deleted):
        if s == 0.0:
            return b
        elif s == 1.0:
            return {}
        else:
            d = changed
            if inserted:
                d[insert] = inserted
            if deleted:
                d[delete] = [pos for pos, value in deleted]
            return d

    def emit_dict_diff(self, a, b, s, added, changed, removed):
        if s == 0.0:
            return b
        elif s == 1.0:
            return {}
        else:
            d = {}
            if added:
                d[insert] = added
            if changed:
                d[update] = changed
            if removed:
                d[delete] = list(removed.keys())
            return d

    def emit_value_diff(self, a, b, s):
        if s == 1.0:
            return {}
        else:
            return b