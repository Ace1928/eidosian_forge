import sys
import json
from .symbols import *
from .symbols import Symbol
class SymmetricJsonDiffSyntax(object):

    def emit_set_diff(self, a, b, s, added, removed):
        if s == 0.0 or len(removed) == len(a):
            return [a, b]
        else:
            d = {}
            if added:
                d[add] = added
            if removed:
                d[discard] = removed
            return d

    def emit_list_diff(self, a, b, s, inserted, changed, deleted):
        if s == 0.0:
            return [a, b]
        elif s == 1.0:
            return {}
        else:
            d = changed
            if inserted:
                d[insert] = inserted
            if deleted:
                d[delete] = deleted
            return d

    def emit_dict_diff(self, a, b, s, added, changed, removed):
        if s == 0.0:
            return [a, b]
        elif s == 1.0:
            return {}
        else:
            d = changed
            if added:
                d[insert] = added
            if removed:
                d[delete] = removed
            return d

    def emit_value_diff(self, a, b, s):
        if s == 1.0:
            return {}
        else:
            return [a, b]

    def patch(self, a, d):
        if isinstance(d, list):
            _, b = d
            return b
        elif isinstance(d, dict):
            if not d:
                return a
            if isinstance(a, dict):
                a = dict(a)
                for k, v in d.items():
                    if k is delete:
                        for kdel, _ in v.items():
                            del a[kdel]
                    elif k is insert:
                        for kk, vv in v.items():
                            a[kk] = vv
                    else:
                        a[k] = self.patch(a[k], v)
                return a
            elif isinstance(a, (list, tuple)):
                original_type = type(a)
                a = list(a)
                if delete in d:
                    for pos, value in d[delete]:
                        a.pop(pos)
                if insert in d:
                    for pos, value in d[insert]:
                        a.insert(pos, value)
                for k, v in d.items():
                    if k is not delete and k is not insert:
                        k = int(k)
                        a[k] = self.patch(a[k], v)
                if original_type is not list:
                    a = original_type(a)
                return a
            elif isinstance(a, set):
                a = set(a)
                if discard in d:
                    for x in d[discard]:
                        a.discard(x)
                if add in d:
                    for x in d[add]:
                        a.add(x)
                return a
        raise Exception('Invalid symmetric diff')

    def unpatch(self, b, d):
        if isinstance(d, list):
            a, _ = d
            return a
        elif isinstance(d, dict):
            if not d:
                return b
            if isinstance(b, dict):
                b = dict(b)
                for k, v in d.items():
                    if k is delete:
                        for kk, vv in v.items():
                            b[kk] = vv
                    elif k is insert:
                        for kk, vv in v.items():
                            del b[kk]
                    else:
                        b[k] = self.unpatch(b[k], v)
                return b
            elif isinstance(b, (list, tuple)):
                original_type = type(b)
                b = list(b)
                for k, v in d.items():
                    if k is not delete and k is not insert:
                        k = int(k)
                        b[k] = self.unpatch(b[k], v)
                if insert in d:
                    for pos, value in reversed(d[insert]):
                        b.pop(pos)
                if delete in d:
                    for pos, value in reversed(d[delete]):
                        b.insert(pos, value)
                if original_type is not list:
                    b = original_type(b)
                return b
            elif isinstance(b, set):
                b = set(b)
                if discard in d:
                    for x in d[discard]:
                        b.add(x)
                if add in d:
                    for x in d[add]:
                        b.discard(x)
                return b
        raise Exception('Invalid symmetric diff')