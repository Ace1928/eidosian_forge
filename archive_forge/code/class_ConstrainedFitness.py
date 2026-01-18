import sys
from copy import deepcopy
from functools import partial
from operator import mul, truediv
class ConstrainedFitness(Fitness):

    def __init__(self, values=(), constraint_violation=None):
        super(ConstrainedFitness, self).__init__(values)
        self.constraint_violation = constraint_violation

    @Fitness.values.deleter
    def values(self):
        self.wvalues = ()
        self.constraint_violation = None

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        self_violates_constraints = _violates_constraint(self)
        other_violates_constraints = _violates_constraint(other)
        if self_violates_constraints and other_violates_constraints:
            return True
        elif self_violates_constraints:
            return True
        elif other_violates_constraints:
            return False
        return self.wvalues <= other.wvalues

    def __lt__(self, other):
        self_violates_constraints = _violates_constraint(self)
        other_violates_constraints = _violates_constraint(other)
        if self_violates_constraints and other_violates_constraints:
            return False
        elif self_violates_constraints:
            return True
        elif other_violates_constraints:
            return False
        return self.wvalues < other.wvalues

    def __eq__(self, other):
        self_violates_constraints = _violates_constraint(self)
        other_violates_constraints = _violates_constraint(other)
        if self_violates_constraints and other_violates_constraints:
            return True
        elif self_violates_constraints:
            return False
        elif other_violates_constraints:
            return False
        return self.wvalues == other.wvalues

    def __ne__(self, other):
        return not self.__eq__(other)

    def dominates(self, other):
        self_violates_constraints = _violates_constraint(self)
        other_violates_constraints = _violates_constraint(other)
        if self_violates_constraints and other_violates_constraints:
            return False
        elif self_violates_constraints:
            return False
        elif other_violates_constraints:
            return True
        return super(ConstrainedFitness, self).dominates(other)

    def __str__(self):
        """Return the values of the Fitness object."""
        return str((self.values if self.valid else tuple(), self.constraint_violation))

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return '%s.%s(%r, %r)' % (self.__module__, self.__class__.__name__, self.values if self.valid else tuple(), self.constraint_violation)