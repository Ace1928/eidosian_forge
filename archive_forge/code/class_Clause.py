import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
class Clause(list):

    def __init__(self, data):
        list.__init__(self, data)
        self._is_tautology = None
        self._parents = None

    def unify(self, other, bindings=None, used=None, skipped=None, debug=False):
        """
        Attempt to unify this Clause with the other, returning a list of
        resulting, unified, Clauses.

        :param other: ``Clause`` with which to unify
        :param bindings: ``BindingDict`` containing bindings that should be used
            during the unification
        :param used: tuple of two lists of atoms.  The first lists the
            atoms from 'self' that were successfully unified with atoms from
            'other'.  The second lists the atoms from 'other' that were successfully
            unified with atoms from 'self'.
        :param skipped: tuple of two ``Clause`` objects.  The first is a list of all
            the atoms from the 'self' Clause that have not been unified with
            anything on the path.  The second is same thing for the 'other' Clause.
        :param debug: bool indicating whether debug statements should print
        :return: list containing all the resulting ``Clause`` objects that could be
            obtained by unification
        """
        if bindings is None:
            bindings = BindingDict()
        if used is None:
            used = ([], [])
        if skipped is None:
            skipped = ([], [])
        if isinstance(debug, bool):
            debug = DebugObject(debug)
        newclauses = _iterate_first(self, other, bindings, used, skipped, _complete_unify_path, debug)
        subsumed = []
        for i, c1 in enumerate(newclauses):
            if i not in subsumed:
                for j, c2 in enumerate(newclauses):
                    if i != j and j not in subsumed and c1.subsumes(c2):
                        subsumed.append(j)
        result = []
        for i in range(len(newclauses)):
            if i not in subsumed:
                result.append(newclauses[i])
        return result

    def isSubsetOf(self, other):
        """
        Return True iff every term in 'self' is a term in 'other'.

        :param other: ``Clause``
        :return: bool
        """
        for a in self:
            if a not in other:
                return False
        return True

    def subsumes(self, other):
        """
        Return True iff 'self' subsumes 'other', this is, if there is a
        substitution such that every term in 'self' can be unified with a term
        in 'other'.

        :param other: ``Clause``
        :return: bool
        """
        negatedother = []
        for atom in other:
            if isinstance(atom, NegatedExpression):
                negatedother.append(atom.term)
            else:
                negatedother.append(-atom)
        negatedotherClause = Clause(negatedother)
        bindings = BindingDict()
        used = ([], [])
        skipped = ([], [])
        debug = DebugObject(False)
        return len(_iterate_first(self, negatedotherClause, bindings, used, skipped, _subsumes_finalize, debug)) > 0

    def __getslice__(self, start, end):
        return Clause(list.__getslice__(self, start, end))

    def __sub__(self, other):
        return Clause([a for a in self if a not in other])

    def __add__(self, other):
        return Clause(list.__add__(self, other))

    def is_tautology(self):
        """
        Self is a tautology if it contains ground terms P and -P.  The ground
        term, P, must be an exact match, ie, not using unification.
        """
        if self._is_tautology is not None:
            return self._is_tautology
        for i, a in enumerate(self):
            if not isinstance(a, EqualityExpression):
                j = len(self) - 1
                while j > i:
                    b = self[j]
                    if isinstance(a, NegatedExpression):
                        if a.term == b:
                            self._is_tautology = True
                            return True
                    elif isinstance(b, NegatedExpression):
                        if a == b.term:
                            self._is_tautology = True
                            return True
                    j -= 1
        self._is_tautology = False
        return False

    def free(self):
        return reduce(operator.or_, (atom.free() | atom.constants() for atom in self))

    def replace(self, variable, expression):
        """
        Replace every instance of variable with expression across every atom
        in the clause

        :param variable: ``Variable``
        :param expression: ``Expression``
        """
        return Clause([atom.replace(variable, expression) for atom in self])

    def substitute_bindings(self, bindings):
        """
        Replace every binding

        :param bindings: A list of tuples mapping Variable Expressions to the
            Expressions to which they are bound.
        :return: ``Clause``
        """
        return Clause([atom.substitute_bindings(bindings) for atom in self])

    def __str__(self):
        return '{' + ', '.join(('%s' % item for item in self)) + '}'

    def __repr__(self):
        return '%s' % self