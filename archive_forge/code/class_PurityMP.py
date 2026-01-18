from typing import Sequence, Optional
import pennylane as qml
from pennylane.wires import Wires
from .measurements import StateMeasurement, Purity
class PurityMP(StateMeasurement):
    """Measurement process that computes the purity of the system prior to measurement.

    Please refer to :func:`purity` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
        id (str): custom label given to a measurement instance, can be useful for some
            applications where the instance has to be identified
    """

    def __init__(self, wires: Wires, id: Optional[str]=None):
        super().__init__(wires=wires, id=id)

    @property
    def return_type(self):
        return Purity

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        if not shots.has_partitioned_shots:
            return ()
        num_shot_elements = sum((s.copies for s in shots.shot_vector))
        return tuple((() for _ in range(num_shot_elements)))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        wire_map = dict(zip(wire_order, list(range(len(wire_order)))))
        indices = [wire_map[w] for w in self.wires]
        state = qml.math.dm_from_state_vector(state)
        return qml.math.purity(state, indices=indices, c_dtype=state.dtype)