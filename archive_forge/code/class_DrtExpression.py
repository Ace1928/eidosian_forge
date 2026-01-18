import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtExpression:
    """
    This is the base abstract DRT Expression from which every DRT
    Expression extends.
    """
    _drt_parser = DrtParser()

    @classmethod
    def fromstring(cls, s):
        return cls._drt_parser.parse(s)

    def applyto(self, other):
        return DrtApplicationExpression(self, other)

    def __neg__(self):
        return DrtNegatedExpression(self)

    def __and__(self, other):
        return NotImplemented

    def __or__(self, other):
        assert isinstance(other, DrtExpression)
        return DrtOrExpression(self, other)

    def __gt__(self, other):
        assert isinstance(other, DrtExpression)
        if isinstance(self, DRS):
            return DRS(self.refs, self.conds, other)
        if isinstance(self, DrtConcatenation):
            return DrtConcatenation(self.first, self.second, other)
        raise Exception('Antecedent of implication must be a DRS')

    def equiv(self, other, prover=None):
        """
        Check for logical equivalence.
        Pass the expression (self <-> other) to the theorem prover.
        If the prover says it is valid, then the self and other are equal.

        :param other: an ``DrtExpression`` to check equality against
        :param prover: a ``nltk.inference.api.Prover``
        """
        assert isinstance(other, DrtExpression)
        f1 = self.simplify().fol()
        f2 = other.simplify().fol()
        return f1.equiv(f2, prover)

    @property
    def type(self):
        raise AttributeError("'%s' object has no attribute 'type'" % self.__class__.__name__)

    def typecheck(self, signature=None):
        raise NotImplementedError()

    def __add__(self, other):
        return DrtConcatenation(self, other, None)

    def get_refs(self, recursive=False):
        """
        Return the set of discourse referents in this DRS.
        :param recursive: bool Also find discourse referents in subterms?
        :return: list of ``Variable`` objects
        """
        raise NotImplementedError()

    def is_pronoun_function(self):
        """Is self of the form "PRO(x)"?"""
        return isinstance(self, DrtApplicationExpression) and isinstance(self.function, DrtAbstractVariableExpression) and (self.function.variable.name == DrtTokens.PRONOUN) and isinstance(self.argument, DrtIndividualVariableExpression)

    def make_EqualityExpression(self, first, second):
        return DrtEqualityExpression(first, second)

    def make_VariableExpression(self, variable):
        return DrtVariableExpression(variable)

    def resolve_anaphora(self):
        return resolve_anaphora(self)

    def eliminate_equality(self):
        return self.visit_structured(lambda e: e.eliminate_equality(), self.__class__)

    def pretty_format(self):
        """
        Draw the DRS
        :return: the pretty print string
        """
        return '\n'.join(self._pretty())

    def pretty_print(self):
        print(self.pretty_format())

    def draw(self):
        DrsDrawer(self).draw()