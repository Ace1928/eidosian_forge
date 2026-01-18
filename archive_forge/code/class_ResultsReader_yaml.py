from pyomo.opt.base import results
from pyomo.opt.base.formats import ResultsFormat
from pyomo.opt import SolverResults
@results.ReaderFactory.register(str(ResultsFormat.yaml))
class ResultsReader_yaml(results.AbstractResultsReader):
    """
    Class that reads in a *.yml file and generates a
    SolverResults object.
    """

    def __init__(self):
        results.AbstractResultsReader.__init__(self, ResultsFormat.yaml)

    def __call__(self, filename, res=None, soln=None, suffixes=[]):
        """
        Parse a *.results file
        """
        if res is None:
            res = SolverResults()
        res.read(filename, using_yaml=True)
        return res