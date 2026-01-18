from sys import version_info as _swig_python_version_info
import weakref
def TabuSearch(self, maximize, objective, step, vars, keep_tenure, forbid_tenure, tabu_factor):
    """
        MetaHeuristics which try to get the search out of local optima.
        Creates a Tabu Search monitor.
        In the context of local search the behavior is similar to MakeOptimize(),
        creating an objective in a given sense. The behavior differs once a local
        optimum is reached: thereafter solutions which degrade the value of the
        objective are allowed if they are not "tabu". A solution is "tabu" if it
        doesn't respect the following rules:
        - improving the best solution found so far
        - variables in the "keep" list must keep their value, variables in the
        "forbid" list must not take the value they have in the list.
        Variables with new values enter the tabu lists after each new solution
        found and leave the lists after a given number of iterations (called
        tenure). Only the variables passed to the method can enter the lists.
        The tabu criterion is softened by the tabu factor which gives the number
        of "tabu" violations which is tolerated; a factor of 1 means no violations
        allowed; a factor of 0 means all violations are allowed.
        """
    return _pywrapcp.Solver_TabuSearch(self, maximize, objective, step, vars, keep_tenure, forbid_tenure, tabu_factor)