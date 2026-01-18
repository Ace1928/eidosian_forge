from ase.ga import get_raw_score
class RawScoreConvergence(Convergence):
    """Returns True if the supplied max_raw_score has been reached"""

    def __init__(self, population_instance, max_raw_score, eps=0.001):
        Convergence.__init__(self, population_instance)
        self.max_raw_score = max_raw_score
        self.eps = eps

    def converged(self):
        cur_pop = self.pop.get_current_population()
        if abs(get_raw_score(cur_pop[0]) - self.max_raw_score) <= self.eps:
            return True
        return False