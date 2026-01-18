import abc
import enum
from typing import Sequence, Dict, Optional, Mapping, NoReturn, List, Tuple
import os
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.errors import ApplicationError
from pyomo.common.deprecation import deprecation_warning
from pyomo.opt.results.results_ import SolverResults as LegacySolverResults
from pyomo.opt.results.solution import Solution as LegacySolution
from pyomo.core.kernel.objective import minimize
from pyomo.core.base import SymbolMap
from pyomo.core.base.label import NumericLabeler
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.config import SolverConfig, PersistentSolverConfig
from pyomo.contrib.solver.util import get_objective
from pyomo.contrib.solver.results import (
class SolverBase(abc.ABC):
    """
    This base class defines the methods required for all solvers:
        - available: Determines whether the solver is able to be run, combining both whether it can be found on the system and if the license is valid.
        - solve: The main method of every solver
        - version: The version of the solver
        - is_persistent: Set to false for all non-persistent solvers.

    Additionally, solvers should have a :attr:`config<SolverBase.config>` attribute that
    inherits from one of :class:`SolverConfig<pyomo.contrib.solver.config.SolverConfig>`,
    :class:`BranchAndBoundConfig<pyomo.contrib.solver.config.BranchAndBoundConfig>`,
    :class:`PersistentSolverConfig<pyomo.contrib.solver.config.PersistentSolverConfig>`, or
    :class:`PersistentBranchAndBoundConfig<pyomo.contrib.solver.config.PersistentBranchAndBoundConfig>`.
    """
    CONFIG = SolverConfig()

    def __init__(self, **kwds) -> None:
        if 'name' in kwds:
            self.name = kwds['name']
            kwds.pop('name')
        else:
            self.name = type(self).__name__.lower()
        self.config = self.CONFIG(value=kwds)

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        """Exit statement - enables `with` statements."""

    class Availability(enum.IntEnum):
        """
        Class to capture different statuses in which a solver can exist in
        order to record its availability for use.
        """
        FullLicense = 2
        LimitedLicense = 1
        NotFound = 0
        BadVersion = -1
        BadLicense = -2
        NeedsCompiledExtension = -3

        def __bool__(self):
            return self._value_ > 0

        def __format__(self, format_spec):
            return format(self.name, format_spec)

        def __str__(self):
            return self.name

    @document_kwargs_from_configdict(CONFIG)
    @abc.abstractmethod
    def solve(self, model: _BlockData, **kwargs) -> Results:
        """
        Solve a Pyomo model.

        Parameters
        ----------
        model: _BlockData
            The Pyomo model to be solved
        **kwargs
            Additional keyword arguments (including solver_options - passthrough
            options; delivered directly to the solver (with no validation))

        Returns
        -------
        results: :class:`Results<pyomo.contrib.solver.results.Results>`
            A results object
        """

    @abc.abstractmethod
    def available(self) -> bool:
        """Test if the solver is available on this system.

        Nominally, this will return True if the solver interface is
        valid and can be used to solve problems and False if it cannot.

        Note that for licensed solvers there are a number of "levels" of
        available: depending on the license, the solver may be available
        with limitations on problem size or runtime (e.g., 'demo'
        vs. 'community' vs. 'full').  In these cases, the solver may
        return a subclass of enum.IntEnum, with members that resolve to
        True if the solver is available (possibly with limitations).
        The Enum may also have multiple members that all resolve to
        False indicating the reason why the interface is not available
        (not found, bad license, unsupported version, etc).

        Returns
        -------
        available: SolverBase.Availability
            An enum that indicates "how available" the solver is.
            Note that the enum can be cast to bool, which will
            be True if the solver is runable at all and False
            otherwise.
        """

    @abc.abstractmethod
    def version(self) -> Tuple:
        """
        Returns
        -------
        version: tuple
            A tuple representing the version
        """

    def is_persistent(self) -> bool:
        """
        Returns
        -------
        is_persistent: bool
            True if the solver is a persistent solver.
        """
        return False