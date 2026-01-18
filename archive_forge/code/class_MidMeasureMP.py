import uuid
from typing import Generic, TypeVar, Optional
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from .measurements import MeasurementProcess, MidMeasure
class MidMeasureMP(MeasurementProcess):
    """Mid-circuit measurement.

    This class additionally stores information about unknown measurement outcomes in the qubit model.
    Measurements on a single qubit in the computational basis are assumed.

    Please refer to :func:`measure` for detailed documentation.

    Args:
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        reset (bool): Whether to reset the wire after measurement.
        postselect (Optional[int]): Which basis state to postselect after a mid-circuit
            measurement. None by default. If postselection is requested, only the post-measurement
            state that is used for postselection will be considered in the remaining circuit.
        id (str): Custom label given to a measurement instance.
    """

    def _flatten(self):
        metadata = (('wires', self.raw_wires), ('reset', self.reset), ('id', self.id))
        return ((None, None), metadata)

    def __init__(self, wires: Optional[Wires]=None, reset: Optional[bool]=False, postselect: Optional[int]=None, id: Optional[str]=None):
        self.batch_size = None
        super().__init__(wires=Wires(wires), id=id)
        self.reset = reset
        self.postselect = postselect

    def label(self, decimals=None, base_label=None, cache=None):
        """How the mid-circuit measurement is represented in diagrams and drawings.

        Args:
            decimals=None (Int): If ``None``, no parameters are included. Else,
                how to round the parameters.
            base_label=None (Iterable[str]): overwrite the non-parameter component of the label.
                Must be same length as ``obs`` attribute.
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings
        """
        _label = '┤↗'
        if self.postselect is not None:
            _label += '₁' if self.postselect == 1 else '₀'
        _label += '├' if not self.reset else '│  │0⟩'
        return _label

    @property
    def return_type(self):
        return MidMeasure

    @property
    def samples_computational_basis(self):
        return False

    @property
    def _queue_category(self):
        return '_ops'

    @property
    def hash(self):
        """int: Returns an integer hash uniquely representing the measurement process"""
        fingerprint = (self.__class__.__name__, tuple(self.wires.tolist()), self.id)
        return hash(fingerprint)

    @property
    def data(self):
        """The data of the measurement. Needed to match the Operator API."""
        return []

    @property
    def name(self):
        """The name of the measurement. Needed to match the Operator API."""
        return 'MidMeasureMP'