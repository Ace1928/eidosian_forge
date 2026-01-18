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
class DefaultLayoutPassManager(PassManagerStagePlugin):
    """Plugin class for default layout stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        _given_layout = SetLayout(pass_manager_config.initial_layout)

        def _choose_layout_condition(property_set):
            return not property_set['layout']

        def _layout_not_perfect(property_set):
            """Return ``True`` if the first attempt at layout has been checked and found to be
            imperfect.  In this case, perfection means "does not require any swap routing"."""
            return property_set['is_swap_mapped'] is not None and (not property_set['is_swap_mapped'])

        def _vf2_match_not_found(property_set):
            if property_set['layout'] is None:
                return True
            return property_set['VF2Layout_stop_reason'] is not None and property_set['VF2Layout_stop_reason'] is not VF2LayoutStopReason.SOLUTION_FOUND

        def _swap_mapped(property_set):
            return property_set['final_layout'] is None
        if pass_manager_config.target is None:
            coupling_map = pass_manager_config.coupling_map
        else:
            coupling_map = pass_manager_config.target
        layout = PassManager()
        layout.append(_given_layout)
        if optimization_level == 0:
            layout.append(ConditionalController(TrivialLayout(coupling_map), condition=_choose_layout_condition))
            layout += common.generate_embed_passmanager(coupling_map)
            return layout
        elif optimization_level == 1:
            layout.append(ConditionalController([TrivialLayout(coupling_map), CheckMap(coupling_map)], condition=_choose_layout_condition))
            choose_layout_1 = VF2Layout(coupling_map=pass_manager_config.coupling_map, seed=pass_manager_config.seed_transpiler, call_limit=int(50000.0), properties=pass_manager_config.backend_properties, target=pass_manager_config.target, max_trials=2500)
            layout.append(ConditionalController(choose_layout_1, condition=_layout_not_perfect))
            choose_layout_2 = SabreLayout(coupling_map, max_iterations=2, seed=pass_manager_config.seed_transpiler, swap_trials=5, layout_trials=5, skip_routing=pass_manager_config.routing_method is not None and pass_manager_config.routing_method != 'sabre')
            layout.append(ConditionalController([BarrierBeforeFinalMeasurements('qiskit.transpiler.internal.routing.protection.barrier'), choose_layout_2], condition=_vf2_match_not_found))
        elif optimization_level == 2:
            choose_layout_0 = VF2Layout(coupling_map=pass_manager_config.coupling_map, seed=pass_manager_config.seed_transpiler, call_limit=int(5000000.0), properties=pass_manager_config.backend_properties, target=pass_manager_config.target, max_trials=25000)
            layout.append(ConditionalController(choose_layout_0, condition=_choose_layout_condition))
            choose_layout_1 = SabreLayout(coupling_map, max_iterations=2, seed=pass_manager_config.seed_transpiler, swap_trials=10, layout_trials=10, skip_routing=pass_manager_config.routing_method is not None and pass_manager_config.routing_method != 'sabre')
            layout.append(ConditionalController([BarrierBeforeFinalMeasurements('qiskit.transpiler.internal.routing.protection.barrier'), choose_layout_1], condition=_vf2_match_not_found))
        elif optimization_level == 3:
            choose_layout_0 = VF2Layout(coupling_map=pass_manager_config.coupling_map, seed=pass_manager_config.seed_transpiler, call_limit=int(30000000.0), properties=pass_manager_config.backend_properties, target=pass_manager_config.target, max_trials=250000)
            layout.append(ConditionalController(choose_layout_0, condition=_choose_layout_condition))
            choose_layout_1 = SabreLayout(coupling_map, max_iterations=4, seed=pass_manager_config.seed_transpiler, swap_trials=20, layout_trials=20, skip_routing=pass_manager_config.routing_method is not None and pass_manager_config.routing_method != 'sabre')
            layout.append(ConditionalController([BarrierBeforeFinalMeasurements('qiskit.transpiler.internal.routing.protection.barrier'), choose_layout_1], condition=_vf2_match_not_found))
        else:
            raise TranspilerError(f'Invalid optimization level: {optimization_level}')
        embed = common.generate_embed_passmanager(coupling_map)
        layout.append(ConditionalController(embed.to_flow_controller(), condition=_swap_mapped))
        return layout