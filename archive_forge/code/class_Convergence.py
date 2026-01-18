from ase.ga import get_raw_score
class Convergence:
    """
    Base class for all convergence object to be based on.
    It is necessary to supply the population instance, to be
    able to obtain current and former populations.
    """

    def __init__(self, population_instance):
        self.pop = population_instance
        self.pops = {}

    def converged(self):
        """This function is called to find out if the algorithm
        run has converged, it should return True or False.
        Overwrite this in the inherited class."""
        raise NotImplementedError

    def populate_pops(self, to_gen):
        """Populate the pops dictionary with how the population
        looked after i number of generations."""
        for i in range(to_gen):
            if i not in self.pops.keys():
                self.pops[i] = self.pop.get_population_after_generation(i)