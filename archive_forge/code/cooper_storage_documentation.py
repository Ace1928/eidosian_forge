from nltk.parse import load_parser
from nltk.parse.featurechart import InstantiateVarsChart
from nltk.sem.logic import ApplicationExpression, LambdaExpression, Variable

        Carry out S-Retrieval of binding operators in store. If hack=True,
        serialize the bindop and core as strings and reparse. Ugh.

        Each permutation of the store (i.e. list of binding operators) is
        taken to be a possible scoping of quantifiers. We iterate through the
        binding operators in each permutation, and successively apply them to
        the current term, starting with the core semantic representation,
        working from the inside out.

        Binding operators are of the form::

             bo(\P.all x.(man(x) -> P(x)),z1)
        