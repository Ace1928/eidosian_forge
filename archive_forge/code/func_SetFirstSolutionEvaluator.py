from sys import version_info as _swig_python_version_info
import weakref
def SetFirstSolutionEvaluator(self, evaluator):
    """
        Gets/sets the evaluator used during the search. Only relevant when
        RoutingSearchParameters.first_solution_strategy = EVALUATOR_STRATEGY.
        Takes ownership of evaluator.
        """
    return _pywrapcp.RoutingModel_SetFirstSolutionEvaluator(self, evaluator)