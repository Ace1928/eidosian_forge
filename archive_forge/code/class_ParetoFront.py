from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq
class ParetoFront(HallOfFame):
    """The Pareto front hall of fame contains all the non-dominated individuals
    that ever lived in the population. That means that the Pareto front hall of
    fame can contain an infinity of different individuals.

    :param similar: A function that tells the Pareto front whether or not two
                    individuals are similar, optional.

    The size of the front may become very large if it is used for example on
    a continuous function with a continuous domain. In order to limit the number
    of individuals, it is possible to specify a similarity function that will
    return :data:`True` if the genotype of two individuals are similar. In that
    case only one of the two individuals will be added to the hall of fame. By
    default the similarity function is :func:`operator.eq`.

    Since, the Pareto front hall of fame inherits from the :class:`HallOfFame`,
    it is sorted lexicographically at every moment.
    """

    def __init__(self, similar=eq):
        HallOfFame.__init__(self, None, similar)

    def update(self, population):
        """Update the Pareto front hall of fame with the *population* by adding
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []
            for i, hofer in enumerate(self):
                if not dominates_one and hofer.fitness.dominates(ind.fitness):
                    is_dominated = True
                    break
                elif ind.fitness.dominates(hofer.fitness):
                    dominates_one = True
                    to_remove.append(i)
                elif ind.fitness == hofer.fitness and self.similar(ind, hofer):
                    has_twin = True
                    break
            for i in reversed(to_remove):
                self.remove(i)
            if not is_dominated and (not has_twin):
                self.insert(ind)