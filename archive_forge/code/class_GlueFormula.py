import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
class GlueFormula:

    def __init__(self, meaning, glue, indices=None):
        if not indices:
            indices = set()
        if isinstance(meaning, str):
            self.meaning = Expression.fromstring(meaning)
        elif isinstance(meaning, Expression):
            self.meaning = meaning
        else:
            raise RuntimeError('Meaning term neither string or expression: %s, %s' % (meaning, meaning.__class__))
        if isinstance(glue, str):
            self.glue = linearlogic.LinearLogicParser().parse(glue)
        elif isinstance(glue, linearlogic.Expression):
            self.glue = glue
        else:
            raise RuntimeError('Glue term neither string or expression: %s, %s' % (glue, glue.__class__))
        self.indices = indices

    def applyto(self, arg):
        """self = (\\x.(walk x), (subj -o f))
        arg  = (john        ,  subj)
        returns ((walk john),          f)
        """
        if self.indices & arg.indices:
            raise linearlogic.LinearLogicApplicationException(f"'{self}' applied to '{arg}'.  Indices are not disjoint.")
        else:
            return_indices = self.indices | arg.indices
        try:
            return_glue = linearlogic.ApplicationExpression(self.glue, arg.glue, arg.indices)
        except linearlogic.LinearLogicApplicationException as e:
            raise linearlogic.LinearLogicApplicationException(f"'{self.simplify()}' applied to '{arg.simplify()}'") from e
        arg_meaning_abstracted = arg.meaning
        if return_indices:
            for dep in self.glue.simplify().antecedent.dependencies[::-1]:
                arg_meaning_abstracted = self.make_LambdaExpression(Variable('v%s' % dep), arg_meaning_abstracted)
        return_meaning = self.meaning.applyto(arg_meaning_abstracted)
        return self.__class__(return_meaning, return_glue, return_indices)

    def make_VariableExpression(self, name):
        return VariableExpression(name)

    def make_LambdaExpression(self, variable, term):
        return LambdaExpression(variable, term)

    def lambda_abstract(self, other):
        assert isinstance(other, GlueFormula)
        assert isinstance(other.meaning, AbstractVariableExpression)
        return self.__class__(self.make_LambdaExpression(other.meaning.variable, self.meaning), linearlogic.ImpExpression(other.glue, self.glue))

    def compile(self, counter=None):
        """From Iddo Lev's PhD Dissertation p108-109"""
        if not counter:
            counter = Counter()
        compiled_glue, new_forms = self.glue.simplify().compile_pos(counter, self.__class__)
        return new_forms + [self.__class__(self.meaning, compiled_glue, {counter.get()})]

    def simplify(self):
        return self.__class__(self.meaning.simplify(), self.glue.simplify(), self.indices)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.meaning == other.meaning and (self.glue == other.glue)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return str(self) < str(other)

    def __str__(self):
        assert isinstance(self.indices, set)
        accum = f'{self.meaning} : {self.glue}'
        if self.indices:
            accum += ' : {' + ', '.join((str(index) for index in sorted(self.indices))) + '}'
        return accum

    def __repr__(self):
        return '%s' % self