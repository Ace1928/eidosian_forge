from typing import Sequence, Optional
import pennylane as qml
from pennylane.wires import Wires
from .measurements import StateMeasurement, VnEntropy
class VnEntropyMP(StateMeasurement):
    """Measurement process that computes the Von Neumann entropy of the system prior to measurement.

    Please refer to :func:`vn_entropy` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
        log_base (float): Base for the logarithm.
    """

    def _flatten(self):
        metadata = (('wires', self.raw_wires), ('log_base', self.log_base))
        return ((None, None), metadata)

    def __init__(self, wires: Optional[Wires]=None, id: Optional[str]=None, log_base: Optional[float]=None):
        self.log_base = log_base
        super().__init__(wires=wires, id=id)

    @property
    def hash(self):
        """int: returns an integer hash uniquely representing the measurement process"""
        fingerprint = (self.__class__.__name__, tuple(self.wires.tolist()), self.log_base)
        return hash(fingerprint)

    @property
    def return_type(self):
        return VnEntropy

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        if not shots.has_partitioned_shots:
            return ()
        num_shot_elements = sum((s.copies for s in shots.shot_vector))
        return tuple((() for _ in range(num_shot_elements)))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        state = qml.math.dm_from_state_vector(state)
        return qml.math.vn_entropy(state, indices=self.wires, c_dtype=state.dtype, base=self.log_base)