import abc
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
This class provides a CyIpoptProblemInterface for use
        with the CyIpoptSolver class that can take in an NLP
        as long as it provides vectors as numpy ndarrays and
        matrices as scipy.sparse.coo_matrix objects. This class
        provides the interface between AmplNLP or PyomoNLP objects
        and the CyIpoptSolver
        