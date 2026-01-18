from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import Error
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import VF2Layout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import OptimizeSwapBeforeMeasure
from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
from qiskit.transpiler.passes.optimization import (
from qiskit.transpiler.passes import Depth, Size, FixedPoint, MinimumPoint
from qiskit.transpiler.passes.utils.gates_basis import GatesInBasis
from qiskit.transpiler.passes.synthesis.unitary_synthesis import UnitarySynthesis
from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.circuit.library.standard_gates import (
class BasisTranslatorPassManager(PassManagerStagePlugin):
    """Plugin class for translation stage with :class:`~.BasisTranslator`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        return common.generate_translation_passmanager(pass_manager_config.target, basis_gates=pass_manager_config.basis_gates, method='translator', approximation_degree=pass_manager_config.approximation_degree, coupling_map=pass_manager_config.coupling_map, backend_props=pass_manager_config.backend_properties, unitary_synthesis_method=pass_manager_config.unitary_synthesis_method, unitary_synthesis_plugin_config=pass_manager_config.unitary_synthesis_plugin_config, hls_config=pass_manager_config.hls_config)