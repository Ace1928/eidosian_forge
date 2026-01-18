import abc
from typing import List, Optional, Dict
import stevedore
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager_config import PassManagerConfig
class PassManagerStagePlugin(abc.ABC):
    """A ``PassManagerStagePlugin`` is a plugin interface object for using custom
    stages in :func:`~.transpile`.

    A ``PassManagerStagePlugin`` object can be added to an external package and
    integrated into the :func:`~.transpile` function with an entry point. This
    will enable users to use the output of :meth:`.pass_manager` to implement
    a stage in the compilation process.
    """

    @abc.abstractmethod
    def pass_manager(self, pass_manager_config: PassManagerConfig, optimization_level: Optional[int]=None) -> PassManager:
        """This method is designed to return a :class:`~.PassManager` for the stage this implements

        Args:
            pass_manager_config: A configuration object that defines all the target device
                specifications and any user specified options to :func:`~.transpile` or
                :func:`~.generate_preset_pass_manager`
            optimization_level: The optimization level of the transpilation, if set this
                should be used to set values for any tunable parameters to trade off runtime
                for potential optimization. Valid values should be ``0``, ``1``, ``2``, or ``3``
                and the higher the number the more optimization is expected.
        """
        pass