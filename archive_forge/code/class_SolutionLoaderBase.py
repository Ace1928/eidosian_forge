import abc
from typing import Sequence, Dict, Optional, Mapping, NoReturn
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr import value
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.sol_reader import SolFileData
from pyomo.repn.plugins.nl_writer import NLWriterInfo
from pyomo.core.expr.visitor import replace_expressions
class SolutionLoaderBase(abc.ABC):
    """
    Base class for all future SolutionLoader classes.

    Intent of this class and its children is to load the solution back into the model.
    """

    def load_vars(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> NoReturn:
        """
        Load the solution of the primal variables into the value attribute of the variables.

        Parameters
        ----------
        vars_to_load: list
            The minimum set of variables whose solution should be loaded. If vars_to_load is None, then the solution
            to all primal variables will be loaded. Even if vars_to_load is specified, the values of other
            variables may also be loaded depending on the interface.
        """
        for v, val in self.get_primals(vars_to_load=vars_to_load).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    @abc.abstractmethod
    def get_primals(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> Mapping[_GeneralVarData, float]:
        """
        Returns a ComponentMap mapping variable to var value.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose solution value should be retrieved. If vars_to_load is None,
            then the values for all variables will be retrieved.

        Returns
        -------
        primals: ComponentMap
            Maps variables to solution values
        """

    def get_duals(self, cons_to_load: Optional[Sequence[_GeneralConstraintData]]=None) -> Dict[_GeneralConstraintData, float]:
        """
        Returns a dictionary mapping constraint to dual value.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose duals should be retrieved. If cons_to_load is None, then the duals for all
            constraints will be retrieved.

        Returns
        -------
        duals: dict
            Maps constraints to dual values
        """
        raise NotImplementedError(f'{type(self)} does not support the get_duals method')

    def get_reduced_costs(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> Mapping[_GeneralVarData, float]:
        """
        Returns a ComponentMap mapping variable to reduced cost.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose reduced cost should be retrieved. If vars_to_load is None, then the
            reduced costs for all variables will be loaded.

        Returns
        -------
        reduced_costs: ComponentMap
            Maps variables to reduced costs
        """
        raise NotImplementedError(f'{type(self)} does not support the get_reduced_costs method')