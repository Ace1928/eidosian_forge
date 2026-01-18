import re
import types
import sys
import os.path
import inspect
import base64
import warnings
class LRGeneratedTable(LRTable):

    def __init__(self, grammar, method='LALR', log=None):
        if method not in ['SLR', 'LALR']:
            raise LALRError('Unsupported method %s' % method)
        self.grammar = grammar
        self.lr_method = method
        if not log:
            log = NullLogger()
        self.log = log
        self.lr_action = {}
        self.lr_goto = {}
        self.lr_productions = grammar.Productions
        self.lr_goto_cache = {}
        self.lr0_cidhash = {}
        self._add_count = 0
        self.sr_conflict = 0
        self.rr_conflict = 0
        self.conflicts = []
        self.sr_conflicts = []
        self.rr_conflicts = []
        self.grammar.build_lritems()
        self.grammar.compute_first()
        self.grammar.compute_follow()
        self.lr_parse_table()

    def lr0_closure(self, I):
        self._add_count += 1
        J = I[:]
        didadd = True
        while didadd:
            didadd = False
            for j in J:
                for x in j.lr_after:
                    if getattr(x, 'lr0_added', 0) == self._add_count:
                        continue
                    J.append(x.lr_next)
                    x.lr0_added = self._add_count
                    didadd = True
        return J

    def lr0_goto(self, I, x):
        g = self.lr_goto_cache.get((id(I), x))
        if g:
            return g
        s = self.lr_goto_cache.get(x)
        if not s:
            s = {}
            self.lr_goto_cache[x] = s
        gs = []
        for p in I:
            n = p.lr_next
            if n and n.lr_before == x:
                s1 = s.get(id(n))
                if not s1:
                    s1 = {}
                    s[id(n)] = s1
                gs.append(n)
                s = s1
        g = s.get('$end')
        if not g:
            if gs:
                g = self.lr0_closure(gs)
                s['$end'] = g
            else:
                s['$end'] = gs
        self.lr_goto_cache[id(I), x] = g
        return g

    def lr0_items(self):
        C = [self.lr0_closure([self.grammar.Productions[0].lr_next])]
        i = 0
        for I in C:
            self.lr0_cidhash[id(I)] = i
            i += 1
        i = 0
        while i < len(C):
            I = C[i]
            i += 1
            asyms = {}
            for ii in I:
                for s in ii.usyms:
                    asyms[s] = None
            for x in asyms:
                g = self.lr0_goto(I, x)
                if not g or id(g) in self.lr0_cidhash:
                    continue
                self.lr0_cidhash[id(g)] = len(C)
                C.append(g)
        return C

    def compute_nullable_nonterminals(self):
        nullable = set()
        num_nullable = 0
        while True:
            for p in self.grammar.Productions[1:]:
                if p.len == 0:
                    nullable.add(p.name)
                    continue
                for t in p.prod:
                    if t not in nullable:
                        break
                else:
                    nullable.add(p.name)
            if len(nullable) == num_nullable:
                break
            num_nullable = len(nullable)
        return nullable

    def find_nonterminal_transitions(self, C):
        trans = []
        for stateno, state in enumerate(C):
            for p in state:
                if p.lr_index < p.len - 1:
                    t = (stateno, p.prod[p.lr_index + 1])
                    if t[1] in self.grammar.Nonterminals:
                        if t not in trans:
                            trans.append(t)
        return trans

    def dr_relation(self, C, trans, nullable):
        dr_set = {}
        state, N = trans
        terms = []
        g = self.lr0_goto(C[state], N)
        for p in g:
            if p.lr_index < p.len - 1:
                a = p.prod[p.lr_index + 1]
                if a in self.grammar.Terminals:
                    if a not in terms:
                        terms.append(a)
        if state == 0 and N == self.grammar.Productions[0].prod[0]:
            terms.append('$end')
        return terms

    def reads_relation(self, C, trans, empty):
        rel = []
        state, N = trans
        g = self.lr0_goto(C[state], N)
        j = self.lr0_cidhash.get(id(g), -1)
        for p in g:
            if p.lr_index < p.len - 1:
                a = p.prod[p.lr_index + 1]
                if a in empty:
                    rel.append((j, a))
        return rel

    def compute_lookback_includes(self, C, trans, nullable):
        lookdict = {}
        includedict = {}
        dtrans = {}
        for t in trans:
            dtrans[t] = 1
        for state, N in trans:
            lookb = []
            includes = []
            for p in C[state]:
                if p.name != N:
                    continue
                lr_index = p.lr_index
                j = state
                while lr_index < p.len - 1:
                    lr_index = lr_index + 1
                    t = p.prod[lr_index]
                    if (j, t) in dtrans:
                        li = lr_index + 1
                        while li < p.len:
                            if p.prod[li] in self.grammar.Terminals:
                                break
                            if p.prod[li] not in nullable:
                                break
                            li = li + 1
                        else:
                            includes.append((j, t))
                    g = self.lr0_goto(C[j], t)
                    j = self.lr0_cidhash.get(id(g), -1)
                for r in C[j]:
                    if r.name != p.name:
                        continue
                    if r.len != p.len:
                        continue
                    i = 0
                    while i < r.lr_index:
                        if r.prod[i] != p.prod[i + 1]:
                            break
                        i = i + 1
                    else:
                        lookb.append((j, r))
            for i in includes:
                if i not in includedict:
                    includedict[i] = []
                includedict[i].append((state, N))
            lookdict[state, N] = lookb
        return (lookdict, includedict)

    def compute_read_sets(self, C, ntrans, nullable):
        FP = lambda x: self.dr_relation(C, x, nullable)
        R = lambda x: self.reads_relation(C, x, nullable)
        F = digraph(ntrans, R, FP)
        return F

    def compute_follow_sets(self, ntrans, readsets, inclsets):
        FP = lambda x: readsets[x]
        R = lambda x: inclsets.get(x, [])
        F = digraph(ntrans, R, FP)
        return F

    def add_lookaheads(self, lookbacks, followset):
        for trans, lb in lookbacks.items():
            for state, p in lb:
                if state not in p.lookaheads:
                    p.lookaheads[state] = []
                f = followset.get(trans, [])
                for a in f:
                    if a not in p.lookaheads[state]:
                        p.lookaheads[state].append(a)

    def add_lalr_lookaheads(self, C):
        nullable = self.compute_nullable_nonterminals()
        trans = self.find_nonterminal_transitions(C)
        readsets = self.compute_read_sets(C, trans, nullable)
        lookd, included = self.compute_lookback_includes(C, trans, nullable)
        followsets = self.compute_follow_sets(trans, readsets, included)
        self.add_lookaheads(lookd, followsets)

    def lr_parse_table(self):
        Productions = self.grammar.Productions
        Precedence = self.grammar.Precedence
        goto = self.lr_goto
        action = self.lr_action
        log = self.log
        actionp = {}
        log.info('Parsing method: %s', self.lr_method)
        C = self.lr0_items()
        if self.lr_method == 'LALR':
            self.add_lalr_lookaheads(C)
        st = 0
        for I in C:
            actlist = []
            st_action = {}
            st_actionp = {}
            st_goto = {}
            log.info('')
            log.info('state %d', st)
            log.info('')
            for p in I:
                log.info('    (%d) %s', p.number, p)
            log.info('')
            for p in I:
                if p.len == p.lr_index + 1:
                    if p.name == "S'":
                        st_action['$end'] = 0
                        st_actionp['$end'] = p
                    else:
                        if self.lr_method == 'LALR':
                            laheads = p.lookaheads[st]
                        else:
                            laheads = self.grammar.Follow[p.name]
                        for a in laheads:
                            actlist.append((a, p, 'reduce using rule %d (%s)' % (p.number, p)))
                            r = st_action.get(a)
                            if r is not None:
                                if r > 0:
                                    sprec, slevel = Precedence.get(a, ('right', 0))
                                    rprec, rlevel = Productions[p.number].prec
                                    if slevel < rlevel or (slevel == rlevel and rprec == 'left'):
                                        st_action[a] = -p.number
                                        st_actionp[a] = p
                                        if not slevel and (not rlevel):
                                            log.info('  ! shift/reduce conflict for %s resolved as reduce', a)
                                            self.sr_conflicts.append((st, a, 'reduce'))
                                        Productions[p.number].reduced += 1
                                    elif slevel == rlevel and rprec == 'nonassoc':
                                        st_action[a] = None
                                    elif not rlevel:
                                        log.info('  ! shift/reduce conflict for %s resolved as shift', a)
                                        self.sr_conflicts.append((st, a, 'shift'))
                                elif r < 0:
                                    oldp = Productions[-r]
                                    pp = Productions[p.number]
                                    if oldp.line > pp.line:
                                        st_action[a] = -p.number
                                        st_actionp[a] = p
                                        chosenp, rejectp = (pp, oldp)
                                        Productions[p.number].reduced += 1
                                        Productions[oldp.number].reduced -= 1
                                    else:
                                        chosenp, rejectp = (oldp, pp)
                                    self.rr_conflicts.append((st, chosenp, rejectp))
                                    log.info('  ! reduce/reduce conflict for %s resolved using rule %d (%s)', a, st_actionp[a].number, st_actionp[a])
                                else:
                                    raise LALRError('Unknown conflict in state %d' % st)
                            else:
                                st_action[a] = -p.number
                                st_actionp[a] = p
                                Productions[p.number].reduced += 1
                else:
                    i = p.lr_index
                    a = p.prod[i + 1]
                    if a in self.grammar.Terminals:
                        g = self.lr0_goto(I, a)
                        j = self.lr0_cidhash.get(id(g), -1)
                        if j >= 0:
                            actlist.append((a, p, 'shift and go to state %d' % j))
                            r = st_action.get(a)
                            if r is not None:
                                if r > 0:
                                    if r != j:
                                        raise LALRError('Shift/shift conflict in state %d' % st)
                                elif r < 0:
                                    sprec, slevel = Precedence.get(a, ('right', 0))
                                    rprec, rlevel = Productions[st_actionp[a].number].prec
                                    if slevel > rlevel or (slevel == rlevel and rprec == 'right'):
                                        Productions[st_actionp[a].number].reduced -= 1
                                        st_action[a] = j
                                        st_actionp[a] = p
                                        if not rlevel:
                                            log.info('  ! shift/reduce conflict for %s resolved as shift', a)
                                            self.sr_conflicts.append((st, a, 'shift'))
                                    elif slevel == rlevel and rprec == 'nonassoc':
                                        st_action[a] = None
                                    elif not slevel and (not rlevel):
                                        log.info('  ! shift/reduce conflict for %s resolved as reduce', a)
                                        self.sr_conflicts.append((st, a, 'reduce'))
                                else:
                                    raise LALRError('Unknown conflict in state %d' % st)
                            else:
                                st_action[a] = j
                                st_actionp[a] = p
            _actprint = {}
            for a, p, m in actlist:
                if a in st_action:
                    if p is st_actionp[a]:
                        log.info('    %-15s %s', a, m)
                        _actprint[a, m] = 1
            log.info('')
            not_used = 0
            for a, p, m in actlist:
                if a in st_action:
                    if p is not st_actionp[a]:
                        if not (a, m) in _actprint:
                            log.debug('  ! %-15s [ %s ]', a, m)
                            not_used = 1
                            _actprint[a, m] = 1
            if not_used:
                log.debug('')
            nkeys = {}
            for ii in I:
                for s in ii.usyms:
                    if s in self.grammar.Nonterminals:
                        nkeys[s] = None
            for n in nkeys:
                g = self.lr0_goto(I, n)
                j = self.lr0_cidhash.get(id(g), -1)
                if j >= 0:
                    st_goto[n] = j
                    log.info('    %-30s shift and go to state %d', n, j)
            action[st] = st_action
            actionp[st] = st_actionp
            goto[st] = st_goto
            st += 1

    def write_table(self, tabmodule, outputdir='', signature=''):
        if isinstance(tabmodule, types.ModuleType):
            raise IOError("Won't overwrite existing tabmodule")
        basemodulename = tabmodule.split('.')[-1]
        filename = os.path.join(outputdir, basemodulename) + '.py'
        try:
            f = open(filename, 'w')
            f.write('\n# %s\n# This file is automatically generated. Do not edit.\n_tabversion = %r\n\n_lr_method = %r\n\n_lr_signature = %r\n    ' % (os.path.basename(filename), __tabversion__, self.lr_method, signature))
            smaller = 1
            if smaller:
                items = {}
                for s, nd in self.lr_action.items():
                    for name, v in nd.items():
                        i = items.get(name)
                        if not i:
                            i = ([], [])
                            items[name] = i
                        i[0].append(s)
                        i[1].append(v)
                f.write('\n_lr_action_items = {')
                for k, v in items.items():
                    f.write('%r:([' % k)
                    for i in v[0]:
                        f.write('%r,' % i)
                    f.write('],[')
                    for i in v[1]:
                        f.write('%r,' % i)
                    f.write(']),')
                f.write('}\n')
                f.write('\n_lr_action = {}\nfor _k, _v in _lr_action_items.items():\n   for _x,_y in zip(_v[0],_v[1]):\n      if not _x in _lr_action:  _lr_action[_x] = {}\n      _lr_action[_x][_k] = _y\ndel _lr_action_items\n')
            else:
                f.write('\n_lr_action = { ')
                for k, v in self.lr_action.items():
                    f.write('(%r,%r):%r,' % (k[0], k[1], v))
                f.write('}\n')
            if smaller:
                items = {}
                for s, nd in self.lr_goto.items():
                    for name, v in nd.items():
                        i = items.get(name)
                        if not i:
                            i = ([], [])
                            items[name] = i
                        i[0].append(s)
                        i[1].append(v)
                f.write('\n_lr_goto_items = {')
                for k, v in items.items():
                    f.write('%r:([' % k)
                    for i in v[0]:
                        f.write('%r,' % i)
                    f.write('],[')
                    for i in v[1]:
                        f.write('%r,' % i)
                    f.write(']),')
                f.write('}\n')
                f.write('\n_lr_goto = {}\nfor _k, _v in _lr_goto_items.items():\n   for _x, _y in zip(_v[0], _v[1]):\n       if not _x in _lr_goto: _lr_goto[_x] = {}\n       _lr_goto[_x][_k] = _y\ndel _lr_goto_items\n')
            else:
                f.write('\n_lr_goto = { ')
                for k, v in self.lr_goto.items():
                    f.write('(%r,%r):%r,' % (k[0], k[1], v))
                f.write('}\n')
            f.write('_lr_productions = [\n')
            for p in self.lr_productions:
                if p.func:
                    f.write('  (%r,%r,%d,%r,%r,%d),\n' % (p.str, p.name, p.len, p.func, os.path.basename(p.file), p.line))
                else:
                    f.write('  (%r,%r,%d,None,None,None),\n' % (str(p), p.name, p.len))
            f.write(']\n')
            f.close()
        except IOError as e:
            raise

    def pickle_table(self, filename, signature=''):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        with open(filename, 'wb') as outf:
            pickle.dump(__tabversion__, outf, pickle_protocol)
            pickle.dump(self.lr_method, outf, pickle_protocol)
            pickle.dump(signature, outf, pickle_protocol)
            pickle.dump(self.lr_action, outf, pickle_protocol)
            pickle.dump(self.lr_goto, outf, pickle_protocol)
            outp = []
            for p in self.lr_productions:
                if p.func:
                    outp.append((p.str, p.name, p.len, p.func, os.path.basename(p.file), p.line))
                else:
                    outp.append((str(p), p.name, p.len, None, None, None))
            pickle.dump(outp, outf, pickle_protocol)