from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
class TableauProver(Prover):
    _assume_false = False

    def _prove(self, goal=None, assumptions=None, verbose=False):
        if not assumptions:
            assumptions = []
        result = None
        try:
            agenda = Agenda()
            if goal:
                agenda.put(-goal)
            agenda.put_all(assumptions)
            debugger = Debug(verbose)
            result = self._attempt_proof(agenda, set(), set(), debugger)
        except RuntimeError as e:
            if self._assume_false and str(e).startswith('maximum recursion depth exceeded'):
                result = False
            elif verbose:
                print(e)
            else:
                raise e
        return (result, '\n'.join(debugger.lines))

    def _attempt_proof(self, agenda, accessible_vars, atoms, debug):
        (current, context), category = agenda.pop_first()
        if not current:
            debug.line('AGENDA EMPTY')
            return False
        proof_method = {Categories.ATOM: self._attempt_proof_atom, Categories.PROP: self._attempt_proof_prop, Categories.N_ATOM: self._attempt_proof_n_atom, Categories.N_PROP: self._attempt_proof_n_prop, Categories.APP: self._attempt_proof_app, Categories.N_APP: self._attempt_proof_n_app, Categories.N_EQ: self._attempt_proof_n_eq, Categories.D_NEG: self._attempt_proof_d_neg, Categories.N_ALL: self._attempt_proof_n_all, Categories.N_EXISTS: self._attempt_proof_n_some, Categories.AND: self._attempt_proof_and, Categories.N_OR: self._attempt_proof_n_or, Categories.N_IMP: self._attempt_proof_n_imp, Categories.OR: self._attempt_proof_or, Categories.IMP: self._attempt_proof_imp, Categories.N_AND: self._attempt_proof_n_and, Categories.IFF: self._attempt_proof_iff, Categories.N_IFF: self._attempt_proof_n_iff, Categories.EQ: self._attempt_proof_eq, Categories.EXISTS: self._attempt_proof_some, Categories.ALL: self._attempt_proof_all}[category]
        debug.line((current, context))
        return proof_method(current, context, agenda, accessible_vars, atoms, debug)

    def _attempt_proof_atom(self, current, context, agenda, accessible_vars, atoms, debug):
        if (current, True) in atoms:
            debug.line('CLOSED', 1)
            return True
        if context:
            if isinstance(context.term, NegatedExpression):
                current = current.negate()
            agenda.put(context(current).simplify())
            return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
        else:
            agenda.mark_alls_fresh()
            return self._attempt_proof(agenda, accessible_vars | set(current.args), atoms | {(current, False)}, debug + 1)

    def _attempt_proof_n_atom(self, current, context, agenda, accessible_vars, atoms, debug):
        if (current.term, False) in atoms:
            debug.line('CLOSED', 1)
            return True
        if context:
            if isinstance(context.term, NegatedExpression):
                current = current.negate()
            agenda.put(context(current).simplify())
            return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
        else:
            agenda.mark_alls_fresh()
            return self._attempt_proof(agenda, accessible_vars | set(current.term.args), atoms | {(current.term, True)}, debug + 1)

    def _attempt_proof_prop(self, current, context, agenda, accessible_vars, atoms, debug):
        if (current, True) in atoms:
            debug.line('CLOSED', 1)
            return True
        agenda.mark_alls_fresh()
        return self._attempt_proof(agenda, accessible_vars, atoms | {(current, False)}, debug + 1)

    def _attempt_proof_n_prop(self, current, context, agenda, accessible_vars, atoms, debug):
        if (current.term, False) in atoms:
            debug.line('CLOSED', 1)
            return True
        agenda.mark_alls_fresh()
        return self._attempt_proof(agenda, accessible_vars, atoms | {(current.term, True)}, debug + 1)

    def _attempt_proof_app(self, current, context, agenda, accessible_vars, atoms, debug):
        f, args = current.uncurry()
        for i, arg in enumerate(args):
            if not TableauProver.is_atom(arg):
                ctx = f
                nv = Variable('X%s' % _counter.get())
                for j, a in enumerate(args):
                    ctx = ctx(VariableExpression(nv)) if i == j else ctx(a)
                if context:
                    ctx = context(ctx).simplify()
                ctx = LambdaExpression(nv, ctx)
                agenda.put(arg, ctx)
                return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
        raise Exception('If this method is called, there must be a non-atomic argument')

    def _attempt_proof_n_app(self, current, context, agenda, accessible_vars, atoms, debug):
        f, args = current.term.uncurry()
        for i, arg in enumerate(args):
            if not TableauProver.is_atom(arg):
                ctx = f
                nv = Variable('X%s' % _counter.get())
                for j, a in enumerate(args):
                    ctx = ctx(VariableExpression(nv)) if i == j else ctx(a)
                if context:
                    ctx = context(ctx).simplify()
                ctx = LambdaExpression(nv, -ctx)
                agenda.put(-arg, ctx)
                return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
        raise Exception('If this method is called, there must be a non-atomic argument')

    def _attempt_proof_n_eq(self, current, context, agenda, accessible_vars, atoms, debug):
        if current.term.first == current.term.second:
            debug.line('CLOSED', 1)
            return True
        agenda[Categories.N_EQ].add((current, context))
        current._exhausted = True
        return self._attempt_proof(agenda, accessible_vars | {current.term.first, current.term.second}, atoms, debug + 1)

    def _attempt_proof_d_neg(self, current, context, agenda, accessible_vars, atoms, debug):
        agenda.put(current.term.term, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_all(self, current, context, agenda, accessible_vars, atoms, debug):
        agenda[Categories.EXISTS].add((ExistsExpression(current.term.variable, -current.term.term), context))
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_some(self, current, context, agenda, accessible_vars, atoms, debug):
        agenda[Categories.ALL].add((AllExpression(current.term.variable, -current.term.term), context))
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_and(self, current, context, agenda, accessible_vars, atoms, debug):
        agenda.put(current.first, context)
        agenda.put(current.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_or(self, current, context, agenda, accessible_vars, atoms, debug):
        agenda.put(-current.term.first, context)
        agenda.put(-current.term.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_imp(self, current, context, agenda, accessible_vars, atoms, debug):
        agenda.put(current.term.first, context)
        agenda.put(-current.term.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_or(self, current, context, agenda, accessible_vars, atoms, debug):
        new_agenda = agenda.clone()
        agenda.put(current.first, context)
        new_agenda.put(current.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_imp(self, current, context, agenda, accessible_vars, atoms, debug):
        new_agenda = agenda.clone()
        agenda.put(-current.first, context)
        new_agenda.put(current.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_and(self, current, context, agenda, accessible_vars, atoms, debug):
        new_agenda = agenda.clone()
        agenda.put(-current.term.first, context)
        new_agenda.put(-current.term.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_iff(self, current, context, agenda, accessible_vars, atoms, debug):
        new_agenda = agenda.clone()
        agenda.put(current.first, context)
        agenda.put(current.second, context)
        new_agenda.put(-current.first, context)
        new_agenda.put(-current.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_n_iff(self, current, context, agenda, accessible_vars, atoms, debug):
        new_agenda = agenda.clone()
        agenda.put(current.term.first, context)
        agenda.put(-current.term.second, context)
        new_agenda.put(-current.term.first, context)
        new_agenda.put(current.term.second, context)
        return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1) and self._attempt_proof(new_agenda, accessible_vars, atoms, debug + 1)

    def _attempt_proof_eq(self, current, context, agenda, accessible_vars, atoms, debug):
        agenda.put_atoms(atoms)
        agenda.replace_all(current.first, current.second)
        accessible_vars.discard(current.first)
        agenda.mark_neqs_fresh()
        return self._attempt_proof(agenda, accessible_vars, set(), debug + 1)

    def _attempt_proof_some(self, current, context, agenda, accessible_vars, atoms, debug):
        new_unique_variable = VariableExpression(unique_variable())
        agenda.put(current.term.replace(current.variable, new_unique_variable), context)
        agenda.mark_alls_fresh()
        return self._attempt_proof(agenda, accessible_vars | {new_unique_variable}, atoms, debug + 1)

    def _attempt_proof_all(self, current, context, agenda, accessible_vars, atoms, debug):
        try:
            current._used_vars
        except AttributeError:
            current._used_vars = set()
        if accessible_vars:
            bv_available = accessible_vars - current._used_vars
            if bv_available:
                variable_to_use = list(bv_available)[0]
                debug.line("--> Using '%s'" % variable_to_use, 2)
                current._used_vars |= {variable_to_use}
                agenda.put(current.term.replace(current.variable, variable_to_use), context)
                agenda[Categories.ALL].add((current, context))
                return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
            else:
                debug.line('--> Variables Exhausted', 2)
                current._exhausted = True
                agenda[Categories.ALL].add((current, context))
                return self._attempt_proof(agenda, accessible_vars, atoms, debug + 1)
        else:
            new_unique_variable = VariableExpression(unique_variable())
            debug.line("--> Using '%s'" % new_unique_variable, 2)
            current._used_vars |= {new_unique_variable}
            agenda.put(current.term.replace(current.variable, new_unique_variable), context)
            agenda[Categories.ALL].add((current, context))
            agenda.mark_alls_fresh()
            return self._attempt_proof(agenda, accessible_vars | {new_unique_variable}, atoms, debug + 1)

    @staticmethod
    def is_atom(e):
        if isinstance(e, NegatedExpression):
            e = e.term
        if isinstance(e, ApplicationExpression):
            for arg in e.args:
                if not TableauProver.is_atom(arg):
                    return False
            return True
        elif isinstance(e, AbstractVariableExpression) or isinstance(e, LambdaExpression):
            return True
        else:
            return False