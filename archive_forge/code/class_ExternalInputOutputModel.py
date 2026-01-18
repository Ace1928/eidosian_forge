import numpy as np
import abc
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.environ import Var, Constraint, value
from pyomo.core.base.var import _VarData
from pyomo.common.modeling import unique_component_name
class ExternalInputOutputModel(object, metaclass=abc.ABCMeta):
    """
    This is the base class for building external input output models
    for use with Pyomo and CyIpopt
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def set_inputs(self, input_values):
        """
        This method is called by the solver to set the current values
        for the input variables. The derived class must cache these if
        necessary for any subsequent calls to evaluate_outputs or
        evaluate_derivatives.
        """
        pass

    @abc.abstractmethod
    def evaluate_outputs(self):
        """
        Compute the outputs from the model (using the values
        set in input_values) and return as a numpy array
        """
        pass

    @abc.abstractmethod
    def evaluate_derivatives(self):
        """
        Compute the derivatives of the outputs with respect
        to the inputs (using the values set in input_values).
        This should be a dense matrix with the rows in
        the order of the output variables and the cols in
        the order of the input variables.
        """
        pass