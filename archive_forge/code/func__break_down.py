from functools import reduce
from nltk.parse import load_parser
from nltk.sem.logic import (
from nltk.sem.skolemize import skolemize
def _break_down(self, usr):
    """
        Extract holes, labels, formula fragments and constraints from the hole
        semantics underspecified representation (USR).
        """
    if isinstance(usr, AndExpression):
        self._break_down(usr.first)
        self._break_down(usr.second)
    elif isinstance(usr, ApplicationExpression):
        func, args = usr.uncurry()
        if func.variable.name == Constants.LEQ:
            self.constraints.add(Constraint(args[0], args[1]))
        elif func.variable.name == Constants.HOLE:
            self.holes.add(args[0])
        elif func.variable.name == Constants.LABEL:
            self.labels.add(args[0])
        else:
            label = args[0]
            assert label not in self.fragments
            self.fragments[label] = (func, args[1:])
    else:
        raise ValueError(usr.label())