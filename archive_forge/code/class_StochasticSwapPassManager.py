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
class StochasticSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.StochasticSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        coupling_map_routing = target
        if coupling_map_routing is None:
            coupling_map_routing = coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(optimization_level, pass_manager_config.layout_method, pass_manager_config.initial_layout)
        if optimization_level == 3:
            routing_pass = StochasticSwap(coupling_map_routing, trials=200, seed=seed_transpiler)
        else:
            routing_pass = StochasticSwap(coupling_map_routing, trials=20, seed=seed_transpiler)
        if optimization_level == 0:
            return common.generate_routing_passmanager(routing_pass, target, coupling_map=coupling_map, seed_transpiler=seed_transpiler, use_barrier_before_measurement=True)
        if optimization_level == 1:
            return common.generate_routing_passmanager(routing_pass, target, coupling_map, vf2_call_limit=vf2_call_limit, vf2_max_trials=vf2_max_trials, backend_properties=backend_properties, seed_transpiler=seed_transpiler, check_trivial=True, use_barrier_before_measurement=True)
        if optimization_level in {2, 3}:
            return common.generate_routing_passmanager(routing_pass, target, coupling_map=coupling_map, vf2_call_limit=vf2_call_limit, vf2_max_trials=vf2_max_trials, backend_properties=backend_properties, seed_transpiler=seed_transpiler, use_barrier_before_measurement=True)
        raise TranspilerError(f'Invalid optimization level specified: {optimization_level}')