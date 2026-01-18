import sys
import json
from .symbols import *
from .symbols import Symbol
class JsonDiffer(object):

    class Options(object):
        pass

    def __init__(self, syntax='compact', load=False, dump=False, marshal=False, loader=default_loader, dumper=default_dumper, escape_str='$'):
        self.options = JsonDiffer.Options()
        self.options.syntax = builtin_syntaxes.get(syntax, syntax)
        self.options.load = load
        self.options.dump = dump
        self.options.marshal = marshal
        self.options.loader = loader
        self.options.dumper = dumper
        self.options.escape_str = escape_str
        self._symbol_map = {escape_str + symbol.label: symbol for symbol in _all_symbols_}

    def _list_diff_0(self, C, X, Y):
        i, j = (len(X), len(Y))
        r = []
        while True:
            if i > 0 and j > 0:
                d, s = self._obj_diff(X[i - 1], Y[j - 1])
                if s > 0 and C[i][j] == C[i - 1][j - 1] + s:
                    r.append((0, d, j - 1, s))
                    i, j = (i - 1, j - 1)
                    continue
            if j > 0 and (i == 0 or C[i][j - 1] >= C[i - 1][j]):
                r.append((1, Y[j - 1], j - 1, 0.0))
                j = j - 1
                continue
            if i > 0 and (j == 0 or C[i][j - 1] < C[i - 1][j]):
                r.append((-1, X[i - 1], i - 1, 0.0))
                i = i - 1
                continue
            return reversed(r)

    def _list_diff(self, X, Y):
        m = len(X)
        n = len(Y)
        C = [[0 for j in range(n + 1)] for i in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                _, s = self._obj_diff(X[i - 1], Y[j - 1])
                C[i][j] = max(C[i][j - 1], C[i - 1][j], C[i - 1][j - 1] + s)
        inserted = []
        deleted = []
        changed = {}
        tot_s = 0.0
        for sign, value, pos, s in self._list_diff_0(C, X, Y):
            if sign == 1:
                inserted.append((pos, value))
            elif sign == -1:
                deleted.insert(0, (pos, value))
            elif sign == 0 and s < 1:
                changed[pos] = value
            tot_s += s
        tot_n = len(X) + len(inserted)
        if tot_n == 0:
            s = 1.0
        else:
            s = tot_s / tot_n
        return (self.options.syntax.emit_list_diff(X, Y, s, inserted, changed, deleted), s)

    def _set_diff(self, a, b):
        removed = a.difference(b)
        added = b.difference(a)
        if not removed and (not added):
            return ({}, 1.0)
        ranking = sorted(((self._obj_diff(x, y)[1], x, y) for x in removed for y in added), reverse=True, key=lambda x: x[0])
        r2 = set(removed)
        a2 = set(added)
        n_common = len(a) - len(removed)
        s_common = float(n_common)
        for s, x, y in ranking:
            if x in r2 and y in a2:
                r2.discard(x)
                a2.discard(y)
                s_common += s
                n_common += 1
            if not r2 or not a2:
                break
        n_tot = len(a) + len(added)
        s = s_common / n_tot if n_tot != 0 else 1.0
        return (self.options.syntax.emit_set_diff(a, b, s, added, removed), s)

    def _dict_diff(self, a, b):
        removed = {}
        nremoved = 0
        nadded = 0
        nmatched = 0
        smatched = 0.0
        added = {}
        changed = {}
        for k, v in a.items():
            w = b.get(k, missing)
            if w is missing:
                nremoved += 1
                removed[k] = v
            else:
                nmatched += 1
                d, s = self._obj_diff(v, w)
                if s < 1.0:
                    changed[k] = d
                smatched += 0.5 + 0.5 * s
        for k, v in b.items():
            if k not in a:
                nadded += 1
                added[k] = v
        n_tot = nremoved + nmatched + nadded
        s = smatched / n_tot if n_tot != 0 else 1.0
        return (self.options.syntax.emit_dict_diff(a, b, s, added, changed, removed), s)

    def _obj_diff(self, a, b):
        if a is b:
            return (self.options.syntax.emit_value_diff(a, b, 1.0), 1.0)
        if isinstance(a, dict) and isinstance(b, dict):
            return self._dict_diff(a, b)
        elif isinstance(a, tuple) and isinstance(b, tuple):
            return self._list_diff(a, b)
        elif isinstance(a, list) and isinstance(b, list):
            return self._list_diff(a, b)
        elif isinstance(a, set) and isinstance(b, set):
            return self._set_diff(a, b)
        elif a != b:
            return (self.options.syntax.emit_value_diff(a, b, 0.0), 0.0)
        else:
            return (self.options.syntax.emit_value_diff(a, b, 1.0), 1.0)

    def diff(self, a, b, fp=None):
        if self.options.load:
            a = self.options.loader(a)
            b = self.options.loader(b)
        d, s = self._obj_diff(a, b)
        if self.options.marshal or self.options.dump:
            d = self.marshal(d)
        if self.options.dump:
            return self.options.dumper(d, fp)
        else:
            return d

    def similarity(self, a, b):
        if self.options.load:
            a = self.options.loader(a)
            b = self.options.loader(b)
        d, s = self._obj_diff(a, b)
        return s

    def patch(self, a, d, fp=None):
        if self.options.load:
            a = self.options.loader(a)
            d = self.options.loader(d)
        if self.options.marshal or self.options.load:
            d = self.unmarshal(d)
        b = self.options.syntax.patch(a, d)
        if self.options.dump:
            return self.options.dumper(b, fp)
        else:
            return b

    def unpatch(self, b, d, fp=None):
        if self.options.load:
            b = self.options.loader(b)
            d = self.options.loader(d)
        if self.options.marshal or self.options.load:
            d = self.unmarshal(d)
        a = self.options.syntax.unpatch(b, d)
        if self.options.dump:
            return self.options.dumper(a, fp)
        else:
            return a

    def _unescape(self, x):
        if isinstance(x, string_types):
            sym = self._symbol_map.get(x, None)
            if sym is not None:
                return sym
            if x.startswith(self.options.escape_str):
                return x[1:]
        return x

    def unmarshal(self, d):
        if isinstance(d, dict):
            return {self._unescape(k): self.unmarshal(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple)):
            return type(d)((self.unmarshal(x) for x in d))
        else:
            return self._unescape(d)

    def _escape(self, o):
        if type(o) is Symbol:
            return self.options.escape_str + o.label
        if isinstance(o, string_types) and o.startswith(self.options.escape_str):
            return self.options.escape_str + o
        return o

    def marshal(self, d):
        if isinstance(d, dict):
            return {self._escape(k): self.marshal(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple)):
            return type(d)((self.marshal(x) for x in d))
        else:
            return self._escape(d)